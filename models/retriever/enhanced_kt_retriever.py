# models/retriever/enhanced_kt_retriever.py
import os
import pickle
import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple
import ast
import faiss
import numpy as np
import spacy
import torch
import torch.nn.functional as F
import concurrent.futures
from sentence_transformers import SentenceTransformer

from models.retriever.faiss_filter import DualFAISSRetriever
from utils import graph_processor
from utils import call_llm_api
from utils.logger import logger

try:
    from config import get_config
except ImportError:
    get_config = None


class KTRetriever:
    def __init__(
        self,
        dataset: str,
        json_path: str = None,
        qa_encoder: Optional[SentenceTransformer] = None,
        device: str = "cuda",
        cache_dir: str = "retriever/faiss_cache_new",
        top_k: int = 5,
        recall_paths: int = 2,
        schema_path: str = None,
        mode: str = "agent",
        config=None,
    ):

        if config is None and get_config is not None:
            try:
                config = get_config()
            except Exception:
                config = None

        self.config = config

        if config:
            json_path = json_path or config.get_dataset_config(dataset).graph_output
            device = device if device != "cuda" else config.embeddings.device
            cache_dir = (
                cache_dir
                if cache_dir != "retriever/faiss_cache_new"
                else config.retrieval.cache_dir
            )
            top_k = top_k if top_k != 5 else config.retrieval.top_k
            recall_paths = (
                recall_paths if recall_paths != 2 else config.retrieval.recall_paths
            )
            schema_path = schema_path or config.get_dataset_config(dataset).schema_path
            mode = mode if mode != "agent" else config.triggers.mode
            qa_encoder = qa_encoder or SentenceTransformer(config.embeddings.model_name)

        self.graph = graph_processor.load_graph_from_json(json_path)
        self.qa_encoder = qa_encoder or SentenceTransformer("all-MiniLM-L6-v2")

        self.llm_client = call_llm_api.LLMCompletionCall()

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Warning: CUDA requested but not available, falling back to CPU"
            )
            self.device = "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.dataset = dataset
        self.schema_path = schema_path
        self.recall_paths = recall_paths
        self.mode = mode
        os.makedirs(cache_dir, exist_ok=True)
        self.debug_mode = True

        self.nlp = spacy.load(config.nlp.spacy_model)

        self.faiss_retriever = DualFAISSRetriever(
            dataset,
            self.graph,
            model_name=config.embeddings.model_name,
            cache_dir=cache_dir,
            device=self.device,
        )

        self.node_embedding_cache = {}
        self.triple_embedding_cache = {}
        self.query_embedding_cache = {}
        self.faiss_search_cache = {}
        self.chunk_embedding_cache = {}
        self.chunk_faiss_index = None
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        self.chunk_embeddings_precomputed = False

        self.cache_locks = {
            "node_embedding": threading.RLock(),
            "triple_embedding": threading.RLock(),
            "query_embedding": threading.RLock(),
            "chunk_embedding": threading.RLock(),
        }

        self.node_embeddings_precomputed = False
        self.precompute_lock = threading.Lock()

        self.chunk2id = {}
        chunk_file = f"output/chunks/{self.dataset}.txt"
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "\t" in line:
                            parts = line.split("\t", 1)
                            if (
                                len(parts) == 2
                                and parts[0].startswith("id: ")
                                and parts[1].startswith("Chunk: ")
                            ):
                                chunk_id = parts[0][4:]
                                chunk_text = parts[1][7:]
                                self.chunk2id[chunk_id] = chunk_text
                logger.info(f"Loaded {len(self.chunk2id)} chunks from {chunk_file}")
            except Exception as e:
                logger.error(f"Error loading chunks from {chunk_file}: {e}")
                self.chunk2id = {}

        self._node_text_index = None
        self.use_exact_keyword_matching = True
        self.enable_performance_optimizations = True
        self._node_text_cache = {}

        if self.enable_performance_optimizations:
            try:
                cache_loaded = self._load_node_embedding_cache()
                self._precompute_node_texts()
                self._build_node_text_index()
                self._precompute_chunk_embeddings()

                if cache_loaded:
                    self.node_embeddings_precomputed = True

                    if (
                        not hasattr(self.faiss_retriever, "node_embedding_cache")
                        or not self.faiss_retriever.node_embedding_cache
                    ):
                        self.faiss_retriever.node_embedding_cache = {}
                        for node, embed in self.node_embedding_cache.items():
                            self.faiss_retriever.node_embedding_cache[node] = (
                                embed.clone().detach()
                            )

            except Exception as e:
                logger.exception("Failed during performance-optimization setup")
                self.enable_performance_optimizations = False

    # ------------------------------------------------------------------
    # Device-safe conversion helper
    # ------------------------------------------------------------------
    @staticmethod
    def _as_cpu_numpy(x) -> np.ndarray:
        """Convert tensor/array-like to a contiguous float32 CPU numpy array.

        Safe to call on CUDA tensors that require grad. This is the ONLY
        conversion path that should touch FAISS or any library that
        internally calls ``tensor.numpy()``.
        """
        if isinstance(x, torch.Tensor):
            t = x.detach()
            if t.is_cuda:
                t = t.cpu()
            if t.dtype != torch.float32:
                t = t.float()
            return t.numpy()
        arr = np.asarray(x)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return np.ascontiguousarray(arr)

    def build_indices(self):
        """Build all FAISS indices for efficient retrieval."""
        self.faiss_retriever.build_indices()
        self._precompute_node_embeddings()

    def _get_query_embedding(self, query: str) -> torch.Tensor:
        """Get query embedding on self.device (torch tensor)."""
        query_embed = (
            torch.tensor(self.qa_encoder.encode(query)).float().to(self.device)
        )
        return query_embed

    # ------------------------------------------------------------------
    # Config-driven prompt resolution
    # ------------------------------------------------------------------

    def _resolve_prompt(
        self, category: str, *keys: Optional[str], **kwargs
    ) -> Optional[str]:
        """Try each key against the config's prompt store, in order.

        Returns the first template that resolves and formats cleanly, or
        ``None`` if none of them work (or if there is no config at all).

        This is the *only* place that knows how prompts are looked up.
        Every `generate_*_prompt` method funnels through it, so adding a
        new dataset-specific prompt is a YAML-only change.
        """
        if not self.config or not hasattr(self.config, "get_prompt_formatted"):
            return None
        for key in keys:
            if not key:
                continue
            try:
                prompt = self.config.get_prompt_formatted(category, key, **kwargs)
            except Exception as e:
                logger.debug(
                    f"prompt lookup miss: {category}.{key} "
                    f"({type(e).__name__}: {e})"
                )
                continue
            if prompt:
                logger.debug(f"prompt lookup hit: {category}.{key}")
                return prompt
        return None

    def _key(self) -> Optional[str]:
        """Dataset-specific key (e.g. ``debt_collection``)."""
        return f"{self.dataset}" if self.dataset else "general"

    # ------------------------------------------------------------------
    # Plain retrieval prompt
    # ------------------------------------------------------------------

    def generate_prompt(self, question: str, context: str) -> str:
        """Plain (non-IRCoT) retrieval prompt for the initial answer attempt.

        Resolution chain:
            1. ``prompts.retrieval.<dataset>``
            2. ``prompts.retrieval.general``
            3. Built-in English fallback
        """
        prompt = self._resolve_prompt(
            "retrieval",
            self.dataset,
            "general",
            question=question,
            context=context,
        )
        if prompt is not None:
            return prompt
        logger.warning(
            f"No config prompt at retrieval.{self.dataset}/general; "
            "using built-in fallback."
        )
        return self._builtin_plain_prompt(question, context)

    # ------------------------------------------------------------------
    # IRCoT prompt
    # ------------------------------------------------------------------

    def generate_ircot_prompt(
        self,
        initial_query: str,
        current_query: str,
        context: str,
    ) -> str:
        """IRCoT reasoning prompt.

        Resolution chain:
            1. ``prompts.retrieval.<dataset>_ircot``
            2. ``prompts.retrieval.ircot``
            3. Built-in English fallback

        Whichever template wins MUST keep the literal markers
        ``"FINAL_ANSWER:"`` and ``"The new query is:"`` — the QA loop
        regex-matches them.
        """
        prompt = self._resolve_prompt(
            "retrieval",
            self._key(),
            initial_query=initial_query,
            current_query=current_query,
            context=context,
        )
        
        if prompt is not None:
            return prompt
        logger.warning(
            f"No config prompt at retrieval.{self._key()}/ircot; "
            "using built-in fallback."
        )
        
        return self._builtin_ircot_prompt(
            current_query, context
        )

    # ------------------------------------------------------------------
    # Built-in fallbacks (config-less / misconfigured YAML only)
    # ------------------------------------------------------------------

    @staticmethod
    def _builtin_plain_prompt(question: str, context: str) -> str:
        return (
            "You are an expert knowledge assistant. Your task is to answer "
            "the question based on the provided knowledge context.\n\n"
            "1. Use ONLY the information from the provided knowledge context "
            "and try your best to answer the question.\n"
            "2. If the knowledge is insufficient, reject to answer the question.\n"
            "3. Be precise and concise in your answer.\n"
            "4. For factual questions, provide the specific fact or entity name.\n"
            "5. For temporal questions, provide the specific date, year, or time period.\n\n"
            f"Question: {question}\n\n"
            f"Knowledge Context:\n{context}\n\n"
            "Answer (be specific and direct):\n"
        )

    @staticmethod
    def _builtin_ircot_prompt(
        current_query: str, context: str
    ) -> str:
        return (
            "You are an expert knowledge assistant using iterative retrieval "
            "with chain-of-thought reasoning.\n\n"
            f"Current Question: {current_query}\n\n"
            "Instructions:\n"
            "1. Analyze the current knowledge context and the question\n"
            "2. Think about what information might be missing or unclear\n"
            "3. If you have enough information to answer, in the end of your "
            'response, write "FINAL_ANSWER:" followed by your final answer\n'
            "4. If you need more information, in the end of your response, write "
            'a specific query begin with "The new query is:" to retrieve '
            "additional relevant information\n"
            "5. Be specific and focused in your reasoning\n\n"
            "Your reasoning:\n"
        )

    def _precompute_node_texts(self):
        """Precompute node texts for all nodes."""
        if self._load_node_text_cache():
            return

        start_time = time.time()

        all_nodes = list(self.graph.nodes())
        processed_nodes = 0

        for node in all_nodes:
            try:
                node_text = self._get_node_text(node)
                if node_text and not node_text.startswith("[Error"):
                    self._node_text_cache[node] = node_text
                processed_nodes += 1
            except Exception:
                continue

        end_time = time.time()
        logger.info(
            f"Node texts precomputed for {len(self._node_text_cache)} nodes "
            f"in {end_time - start_time:.2f} seconds"
        )

        try:
            self._save_node_text_cache()
        except Exception as e:
            logger.warning(f"Failed to save node text cache: {type(e).__name__}: {e}")

    def _save_node_text_cache(self):
        """Save node text cache to disk."""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_cache.pkl"
        try:
            if not self._node_text_cache:
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with open(cache_path, "wb") as f:
                pickle.dump(self._node_text_cache, f)

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"Saved node text cache with {len(self._node_text_cache)} entries "
                f"to {cache_path} (size: {file_size} bytes)"
            )
            return True

        except Exception:
            return False

    def _load_node_text_cache(self):
        """Load node text cache from disk."""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_cache.pkl"
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    logger.warning(
                        f"Warning: Cache file too small ({file_size} bytes), likely empty or corrupted"
                    )
                    return False

                with open(cache_path, "rb") as f:
                    self._node_text_cache = pickle.load(f)

                if not self._node_text_cache:
                    logger.warning("Warning: Loaded cache is empty")
                    return False

                if not self._check_text_cache_consistency():
                    logger.warning(
                        "Text cache inconsistent with current graph, will rebuild"
                    )
                    return False

                logger.info(
                    f"Loaded node text cache with {len(self._node_text_cache)} entries "
                    f"from {cache_path} (file size: {file_size} bytes)"
                )
                return True

            except Exception as e:
                logger.error(f"Error loading node text cache: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
                except Exception as e2:
                    logger.warning(
                        f"Failed to remove corrupted cache file {cache_path}: {type(e2).__name__}: {e2}"
                    )
        else:
            logger.warning(f"Cache file not found: {cache_path}")
        return False

    def _check_text_cache_consistency(self):
        """Check if the loaded text cache is consistent with current graph."""
        try:
            current_nodes = set(self.graph.nodes())
            cached_nodes = set(self._node_text_cache.keys())

            missing_nodes = current_nodes - cached_nodes
            if missing_nodes:
                logger.info(
                    f"Text cache missing {len(missing_nodes)} nodes from current graph"
                )
                return False

            extra_nodes = cached_nodes - current_nodes
            if len(extra_nodes) > len(current_nodes) * 0.1:
                logger.warning(
                    f"Text cache has too many extra nodes: {len(extra_nodes)} extra "
                    f"vs {len(current_nodes)} current"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking text cache consistency: {e}")
            return False

    def _precompute_node_embeddings(self):
        """Precompute embeddings for all nodes."""
        with self.precompute_lock:
            if self.node_embeddings_precomputed:
                return

            if self._load_node_embedding_cache():
                self.node_embeddings_precomputed = True
                return

            if (
                hasattr(self.faiss_retriever, "node_embedding_cache")
                and self.faiss_retriever.node_embedding_cache
            ):
                for node, embed in self.faiss_retriever.node_embedding_cache.items():
                    self.node_embedding_cache[node] = embed.clone().detach()
                self.node_embeddings_precomputed = True
                logger.info(
                    f"Successfully loaded {len(self.node_embedding_cache)} node embeddings "
                    f"from faiss_retriever cache"
                )
                self._save_node_embedding_cache()
                return

            logger.warning("No cache found, computing embeddings from scratch...")

            all_nodes = list(self.graph.nodes())
            batch_size = 100
            if self.config:
                batch_size = self.config.embeddings.batch_size * 3

            total_processed = 0
            for i in range(0, len(all_nodes), batch_size):
                batch_nodes = all_nodes[i : i + batch_size]
                batch_texts = []
                valid_nodes = []

                for node in batch_nodes:
                    try:
                        node_text = self._get_node_text(node)
                        if node_text and not node_text.startswith("[Error"):
                            batch_texts.append(node_text)
                            valid_nodes.append(node)
                    except Exception as e:
                        logger.error(f"Error getting text for node {node}: {str(e)}")
                        continue

                if batch_texts:
                    try:
                        batch_embeddings = self.qa_encoder.encode(
                            batch_texts, convert_to_tensor=True
                        )

                        for j, node in enumerate(valid_nodes):
                            self.node_embedding_cache[node] = batch_embeddings[j]
                            total_processed += 1

                    except Exception as e:
                        logger.error(
                            f"Error encoding batch {i // batch_size}: {str(e)}"
                        )
                        for node in valid_nodes:
                            try:
                                node_text = self._get_node_text(node)
                                if node_text and not node_text.startswith("[Error"):
                                    embedding = (
                                        torch.tensor(self.qa_encoder.encode(node_text))
                                        .float()
                                        .to(self.device)
                                    )
                                    self.node_embedding_cache[node] = embedding
                                    total_processed += 1
                            except Exception as e2:
                                logger.error(f"Error encoding node {node}: {str(e2)}")
                                continue

            self.node_embeddings_precomputed = True
            logger.info(
                f"Node embeddings precomputed for {total_processed} nodes "
                f"(cache size: {len(self.node_embedding_cache)})"
            )

            try:
                self._save_node_embedding_cache()
            except Exception as e:
                logger.warning(f"Failed to save node embedding cache: {e}")
                logger.info("Continuing without saving cache...")

            self._cleanup_node_cache()

    def _save_node_embedding_cache(self):
        """Save node embedding cache to disk."""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_embedding_cache.pt"
        try:
            if not self.node_embedding_cache:
                logger.warning("Warning: No node embeddings to save!")
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            cpu_cache = {}
            for node, embed in self.node_embedding_cache.items():
                if embed is not None:
                    try:
                        if hasattr(embed, "detach"):
                            cpu_cache[node] = embed.detach().cpu().numpy()
                        elif isinstance(embed, np.ndarray):
                            cpu_cache[node] = embed
                        else:
                            cpu_cache[node] = np.array(embed)
                    except Exception as e:
                        logger.warning(
                            f"Warning: Failed to convert embedding for node {node}: {e}"
                        )
                        continue

            if not cpu_cache:
                logger.warning("Warning: No valid embeddings to save!")
                return False

            try:
                tensor_cache = {}
                for node, embed_array in cpu_cache.items():
                    if isinstance(embed_array, np.ndarray):
                        tensor_cache[node] = torch.from_numpy(embed_array).float()
                    else:
                        tensor_cache[node] = embed_array

                torch.save(tensor_cache, cache_path)
                logger.info(
                    f"Saved node embedding cache using torch.save with tensor format"
                )
            except Exception as torch_error:
                logger.error(f"torch.save failed: {torch_error}, using numpy.save")
                cache_path_npz = cache_path.replace(".pt", ".npz")
                np.savez_compressed(cache_path_npz, **cpu_cache)
                cache_path = cache_path_npz
                logger.error(f"Saved using numpy.savez_compressed format")

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"Saved node embedding cache with {len(cpu_cache)} entries to "
                f"{cache_path} (size: {file_size} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving node embedding cache: {e}")
            return False

    def _load_node_embedding_cache(self):
        """Load node embedding cache from disk."""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_embedding_cache.pt"
        cache_path_npz = cache_path.replace(".pt", ".npz")

        if os.path.exists(cache_path_npz):
            try:
                file_size = os.path.getsize(cache_path_npz)
                logger.info(
                    f"Loading node embedding cache from {cache_path_npz} "
                    f"(file size: {file_size} bytes)"
                )

                numpy_cache = np.load(cache_path_npz)

                if len(numpy_cache.files) == 0:
                    logger.warning("Warning: Loaded cache is empty")
                    return False

                self.node_embedding_cache.clear()

                for node in numpy_cache.files:
                    try:
                        embed_array = numpy_cache[node]
                        embed_tensor = (
                            torch.from_numpy(embed_array).float().to(self.device)
                        )
                        self.node_embedding_cache[node] = embed_tensor
                    except Exception as e:
                        logger.warning(
                            f"Warning: Failed to load embedding for node {node}: {e}"
                        )
                        continue

                numpy_cache.close()

                if not self._check_embedding_cache_consistency():
                    logger.info(
                        "Embedding cache inconsistent with current graph, will rebuild"
                    )
                    return False

                logger.info(
                    f"Loaded node embedding cache with {len(self.node_embedding_cache)} "
                    f"entries from {cache_path_npz}"
                )
                return True

            except Exception as e:
                logger.error(f"Error loading numpy cache: {e}")

        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    logger.warning(
                        f"Warning: Cache file too small ({file_size} bytes), likely empty or corrupted"
                    )
                    return False

                try:
                    cpu_cache = torch.load(
                        cache_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    cpu_cache = torch.load(cache_path, map_location="cpu")
                except Exception as e:
                    if "numpy.core.multiarray._reconstruct" in str(e):
                        try:
                            import importlib

                            torch_serialization = importlib.import_module(
                                "torch.serialization"
                            )
                            torch_serialization.add_safe_globals(
                                ["numpy.core.multiarray._reconstruct"]
                            )
                            cpu_cache = torch.load(cache_path, map_location="cpu")
                        except Exception:
                            raise e
                    else:
                        raise e

                if not cpu_cache:
                    logger.warning("Warning: Loaded cache is empty")
                    return False

                self.node_embedding_cache.clear()

                for node, embed in cpu_cache.items():
                    if embed is not None:
                        try:
                            if isinstance(embed, np.ndarray):
                                embed_tensor = torch.from_numpy(embed).float()
                            else:
                                embed_tensor = (
                                    embed.cpu() if hasattr(embed, "cpu") else embed
                                )

                            if self.device == "cuda" and torch.cuda.is_available():
                                embed_tensor = embed_tensor.to(self.device)
                            else:
                                embed_tensor = embed_tensor.to("cpu")

                            self.node_embedding_cache[node] = embed_tensor
                        except Exception as e:
                            logger.error(
                                f"Warning: Failed to load embedding for node {node}: {e}"
                            )
                            continue

                if not self._check_embedding_cache_consistency():
                    logger.info(
                        "Embedding cache inconsistent with current graph, will rebuild"
                    )
                    return False

                logger.info(
                    f"Loaded node embedding cache with {len(self.node_embedding_cache)} entries "
                    f"from {cache_path} (file size: {file_size} bytes)"
                )
                return True

            except Exception as e:
                logger.error(f"Error loading node embedding cache: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
                except Exception as e3:
                    logger.warning(
                        f"Failed to remove corrupted cache file {cache_path}: {type(e3).__name__}: {e3}"
                    )
        else:
            logger.info(f"Cache file not found: {cache_path}")
        return False

    def _check_embedding_cache_consistency(self):
        """Check if the loaded embedding cache is consistent with current graph."""
        try:
            current_nodes = set(self.graph.nodes())
            cached_nodes = set(self.node_embedding_cache.keys())

            missing_nodes = current_nodes - cached_nodes
            if missing_nodes:
                logger.info(
                    f"Embedding cache missing {len(missing_nodes)} nodes from current graph"
                )
                return False

            extra_nodes = cached_nodes - current_nodes
            if len(extra_nodes) > len(current_nodes) * 0.1:
                logger.info(
                    f"Embedding cache has too many extra nodes: {len(extra_nodes)} extra "
                    f"vs {len(current_nodes)} current"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking embedding cache consistency: {e}")
            return False

    def _cleanup_node_cache(self):
        """Clean up node embedding cache to save memory."""
        with self.cache_locks["node_embedding"]:
            if len(self.node_embedding_cache) > 5000:
                recent_nodes = list(self.node_embedding_cache.keys())[-5000:]
                self.node_embedding_cache = {
                    k: self.node_embedding_cache[k] for k in recent_nodes
                }

    def retrieve(self, question: str) -> Dict:
        """Perform enhanced two-path retrieval."""
        start_time = time.time()

        question_embed = self._get_query_embedding(question)
        query_time = time.time() - start_time

        all_chunk_ids = set()

        if self.recall_paths == 1:
            path_start = time.time()
            path1_results = self._node_relation_retrieval(question_embed, question)
            path1_time = time.time() - path_start
            logger.info(
                f"Query encoding: {query_time:.3f}s, Path1 retrieval: {path1_time:.3f}s"
            )

            path1_chunk_ids = self._extract_chunk_ids_from_nodes(
                path1_results["top_nodes"]
            )
            all_chunk_ids.update(path1_chunk_ids)

            if "chunk_results" in path1_results and path1_results["chunk_results"]:
                chunk_chunk_ids = set(
                    path1_results["chunk_results"].get("chunk_ids", [])
                )
                all_chunk_ids.update(chunk_chunk_ids)

            limited_chunk_ids = list(all_chunk_ids)

            result = {"path1_results": path1_results, "chunk_ids": limited_chunk_ids}
        else:
            parallel_start = time.time()
            result = self._parallel_dual_path_retrieval(question_embed, question)
            parallel_time = time.time() - parallel_start
            logger.info(
                f"Query encoding: {query_time:.3f}s, Parallel retrieval: {parallel_time:.3f}s"
            )

        return question_embed, result

    def retrieve_with_type_filtering(self, question: str, involved_types: dict = None) -> Dict:
        question_embed = self._get_query_embedding(question)

        if not (involved_types and any(involved_types.get(k, []) for k in ("nodes", "relations", "attributes"))):
            return question_embed, self.retrieve(question)[1]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_filtered = ex.submit(self._type_based_retrieval, question_embed, question, involved_types)
            f_full     = ex.submit(self._node_relation_retrieval, question_embed, question)

            filtered = f_filtered.result()
            full     = f_full.result()

        # Merge: union of triples + union of chunk_results
        merged_path1 = {
            "top_nodes": list({*filtered["path1_results"]["top_nodes"],
                            *full["top_nodes"]}),
            "one_hop_triples": list({*filtered["path1_results"]["one_hop_triples"],
                                    *full["one_hop_triples"]}),
            "chunk_results": full.get("chunk_results"),
        }

        return question_embed, {
            "path1_results": merged_path1,
            "path2_results": filtered["path2_results"],
        }

    def _type_based_retrieval(
        self, question_embed: torch.Tensor, question: str, involved_types: dict
    ) -> Dict:
        """Perform hybrid retrieval: type-filtered node_relation path + original other paths."""
        if self.recall_paths == 1:
            filtered_results = self._type_filtered_node_relation_retrieval(
                question_embed, question, involved_types
            )
            return filtered_results
        else:
            hybrid_results = self._hybrid_type_filtered_retrieval(
                question_embed, question, involved_types
            )
            return hybrid_results

    def _type_filtered_node_relation_retrieval(
        self, question_embed: torch.Tensor, question: str, involved_types: dict
    ) -> Dict:
        """Single path retrieval with type filtering only on node_relation path."""
        target_node_types = involved_types.get("nodes", [])

        type_filtered_nodes = self._filter_nodes_by_schema_type(target_node_types)

        if type_filtered_nodes:
            filtered_node_results = self._similarity_search_on_filtered_nodes(
                question_embed, type_filtered_nodes
            )

            one_hop_triples = self._get_one_hop_triples_from_nodes(
                filtered_node_results["top_nodes"]
            )

            chunk_ids = self._extract_chunk_ids_from_nodes(
                filtered_node_results["top_nodes"]
            )

            result = {
                "path1_results": {
                    "top_nodes": filtered_node_results["top_nodes"],
                    "one_hop_triples": one_hop_triples,
                },
                "chunk_ids": list(chunk_ids),
            }
        else:
            result = self._node_relation_retrieval(question_embed, question)

        return result

    def _hybrid_type_filtered_retrieval(
        self, question_embed: torch.Tensor, question: str, involved_types: dict
    ) -> Dict:
        """Multi-path retrieval: type-filtered node_relation + original other paths."""
        target_node_types = involved_types.get("nodes", [])

        if target_node_types:
            type_filtered_nodes = self._filter_nodes_by_schema_type(target_node_types)
            if type_filtered_nodes:
                path1_results = self._type_filtered_node_relation_path(question_embed, type_filtered_nodes)
            else:
                logger.info("Type filter matched nothing; falling back to full node_relation retrieval.")
                path1_results = self._node_relation_retrieval(question_embed, question)
        else:
            path1_results = self._node_relation_retrieval(question_embed, question)

        path2_results = self._triple_only_retrieval(question_embed)

        result = {
            "path1_results": path1_results,
            "path2_results": path2_results,
        }

        return result

    def _type_filtered_node_relation_path(
        self, question_embed: torch.Tensor, filtered_nodes: list
    ) -> Dict:
        """Execute type-filtered node_relation path."""
        filtered_node_results = self._similarity_search_on_filtered_nodes(
            question_embed, filtered_nodes
        )

        one_hop_triples = self._get_one_hop_triples_from_nodes(
            filtered_node_results["top_nodes"]
        )

        return {
            "top_nodes": filtered_node_results["top_nodes"],
            "one_hop_triples": one_hop_triples,
        }

    def _similarity_search_on_filtered_nodes(
        self, question_embed: torch.Tensor, filtered_nodes: list
    ) -> Dict:
        """Perform similarity search only on filtered nodes.

        NOTE: ``question_embed`` may live on CUDA. FAISS' Python bindings call
        ``ndarray.__array_interface__`` / tensor ``.numpy()`` internally, which
        raises on CUDA tensors. We therefore always pass a CPU float32
        numpy array through ``_as_cpu_numpy``.
        """
        if not filtered_nodes:
            return {"top_nodes": []}

        filtered_node_embeddings = []
        filtered_node_map = {}

        # Build a reverse lookup once — the original loop below was O(N^2).
        node_id_to_orig_idx = {
            node_id: orig_idx
            for orig_idx, node_id in self.faiss_retriever.node_map.items()
        }

        for node_id in filtered_nodes:
            original_idx = node_id_to_orig_idx.get(node_id)
            if original_idx is None:
                continue
            try:
                node_embedding = self.faiss_retriever.node_index.reconstruct(
                    int(original_idx)
                )
            except Exception:
                continue
            filtered_node_embeddings.append(node_embedding)
            filtered_node_map[len(filtered_node_embeddings) - 1] = node_id

        if filtered_node_embeddings:
            filtered_embeddings_array = np.ascontiguousarray(
                np.array(filtered_node_embeddings, dtype="float32")
            )
            temp_index = faiss.IndexFlatIP(filtered_embeddings_array.shape[1])
            temp_index.add(filtered_embeddings_array)

            search_k = min(self.top_k, len(filtered_node_embeddings))

            # === CUDA-safe: convert to CPU numpy before touching FAISS ===
            query_np = self._as_cpu_numpy(question_embed).reshape(1, -1)
            _, indices = temp_index.search(query_np, search_k)

            top_filtered_nodes = [
                filtered_node_map[int(idx)]
                for idx in indices[0]
                if int(idx) in filtered_node_map
            ]
        else:
            top_filtered_nodes = filtered_nodes[: self.top_k]

        return {"top_nodes": top_filtered_nodes}

    def _get_one_hop_triples_from_nodes(self, node_list: list) -> list:
        """Return 1-hop triples as (head_id, relation, tail_id) using graph keys.

        Returning node IDs (not display names) is required downstream:
        `_get_edge_chunk_ids`, `_get_node_text`, and `_get_node_properties` all
        look nodes up by their graph key.  Display names are applied later by
        `_format_scored_triples` via `_get_node_text`.
        """
        one_hop_triples = []
        node_set = set(node_list)
        for u, v, data in self.graph.edges(data=True):
            if u in node_set or v in node_set:
                relation = data.get("relation", "")
                if relation:
                    one_hop_triples.append((u, relation, v))
        return one_hop_triples[: self.top_k]

    def _filter_nodes_by_schema_type(self, target_types: list) -> list:
        """Restrict to nodes whose declared class is in `target_types`.

        kt_gen stores the node type under ``properties.class`` (the pydantic
        model class name, e.g. ``MasterStage``); older graphs may use
        ``schema_type``.  Matching either keeps both layouts working.
        """
        if not target_types:
            return list(self.graph.nodes())

        wanted = set(target_types)
        filtered = []
        for node_id, node_data in self.graph.nodes(data=True):
            props = node_data.get("properties", {}) or {}
            node_class = props.get("class") or props.get("schema_type") or ""
            if node_class in wanted:
                filtered.append(node_id)
        return filtered

    def _get_node_name(self, node_id: str) -> str:
        """Get the name property of a node."""
        node_data = self.graph.nodes.get(node_id, {})
        properties = node_data.get("properties", {})
        return properties.get("name", node_id)

    def _parallel_dual_path_retrieval(
        self, question_embed: torch.Tensor, question: str
    ) -> Dict:
        all_chunk_ids = set()
        start_time = time.time()

        max_workers = 4
        if self.config:
            max_workers = self.config.retrieval.faiss.max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            path1_future = executor.submit(
                self._node_relation_retrieval, question_embed, question
            )
            path2_future = executor.submit(self._triple_only_retrieval, question_embed)

            path1_results = path1_future.result()
            path2_results = path2_future.result()

        start_time = time.time()

        path1_chunk_ids = self._extract_chunk_ids_from_nodes(path1_results["top_nodes"])
        path2_chunk_ids = self._extract_chunk_ids_from_triples(
            path2_results["scored_triples"]
        )

        path3_chunk_ids = set()
        if "chunk_results" in path1_results and path1_results["chunk_results"]:
            path3_chunk_ids = set(path1_results["chunk_results"].get("chunk_ids", []))

        all_chunk_ids.update(path1_chunk_ids)
        all_chunk_ids.update(path2_chunk_ids)
        all_chunk_ids.update(path3_chunk_ids)

        limited_chunk_ids = list(all_chunk_ids)[: self.top_k]

        end_time = time.time()
        logger.info(f"Time taken to extract chunk IDs: {end_time - start_time} seconds")
        return {
            "path1_results": path1_results,
            "path2_results": path2_results,
            "chunk_ids": limited_chunk_ids,
        }

    def _execute_retrieval_strategies_parallel(
        self, question_embed: torch.Tensor, question: str, q_embed
    ) -> Dict:
        """Execute multiple retrieval strategies in parallel."""
        results = {
            "faiss_nodes": [],
            "faiss_relations": [],
            "keyword_nodes": [],
            "path_triples": [],
            "keywords": [],
        }
        max_workers = 4
        if self.config:
            max_workers = self.config.retrieval.faiss.max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            faiss_node_future = executor.submit(
                self._faiss_node_search, q_embed, min(self.top_k * 3, 50)
            )
            faiss_relation_future = executor.submit(
                self._faiss_relation_search, q_embed, self.top_k
            )

            if question:
                keyword_future = executor.submit(
                    self._keyword_strategy, question, question_embed
                )
            else:
                keyword_future = None

            if question:
                path_future = executor.submit(
                    self._path_strategy, question, question_embed
                )
            else:
                path_future = None

            try:
                results["faiss_nodes"] = faiss_node_future.result()
            except Exception as e:
                logger.error(f"FAISS node search failed: {e}")

            try:
                results["faiss_relations"] = faiss_relation_future.result()
            except Exception as e:
                logger.error(f"FAISS relation search failed: {e}")

            if keyword_future:
                try:
                    keyword_results = keyword_future.result()
                    results["keyword_nodes"] = keyword_results.get("nodes", [])
                    results["keywords"] = keyword_results.get("keywords", [])
                except Exception as e:
                    logger.error(f"Keyword strategy failed: {e}")

            if path_future:
                try:
                    results["path_triples"] = path_future.result()
                except Exception as e:
                    logger.error(f"Path strategy failed: {e}")

        return results

    def _faiss_node_search(self, q_embed, search_k: int) -> List[str]:
        """Execute FAISS node search with caching. CUDA-safe."""
        q_np = self._as_cpu_numpy(q_embed)
        search_key = f"node_search_{hash(q_np.tobytes())}_{search_k}"

        if (
            hasattr(self, "faiss_search_cache")
            and search_key in self.faiss_search_cache
        ):
            D_nodes, I_nodes = self.faiss_search_cache[search_key]
        else:
            D_nodes, I_nodes = self.faiss_retriever.node_index.search(
                q_np.reshape(1, -1), search_k
            )
            if not hasattr(self, "faiss_search_cache"):
                self.faiss_search_cache = {}
            self.faiss_search_cache[search_key] = (D_nodes, I_nodes)

        candidate_nodes = []
        for idx in I_nodes[0]:
            if idx == -1:
                continue
            try:
                node_id = self.faiss_retriever.node_map[str(idx)]
                if node_id in self.graph.nodes:
                    candidate_nodes.append(node_id)
            except KeyError:
                continue

        return candidate_nodes

    def _faiss_relation_search(self, q_embed, top_k: int) -> List[str]:
        """Execute FAISS relation search with caching. CUDA-safe."""
        q_np = self._as_cpu_numpy(q_embed)
        search_key = f"relation_search_{hash(q_np.tobytes())}_{top_k}"

        if (
            hasattr(self, "faiss_search_cache")
            and search_key in self.faiss_search_cache
        ):
            D_relations, I_relations = self.faiss_search_cache[search_key]
        else:
            D_relations, I_relations = self.faiss_retriever.relation_index.search(
                q_np.reshape(1, -1), top_k
            )
            if not hasattr(self, "faiss_search_cache"):
                self.faiss_search_cache = {}
            self.faiss_search_cache[search_key] = (D_relations, I_relations)

        relations = []
        for idx in I_relations[0]:
            if idx == -1:
                continue
            try:
                relation = self.faiss_retriever.relation_map[str(idx)]
                relations.append(relation)
            except KeyError:
                continue

        return relations

    def _keyword_strategy(self, question: str, question_embed: torch.Tensor) -> Dict:
        """Execute keyword extraction and search strategy."""
        keywords = self._extract_query_keywords(question)
        keyword_nodes = self._keyword_based_node_search(keywords)

        return {"keywords": keywords, "nodes": keyword_nodes}

    def _path_strategy(self, question: str):
        """Execute path-based search strategy."""
        self._extract_query_keywords(question)
        return

    def _node_relation_retrieval(
        self, question_embed: torch.Tensor, question: str = ""
    ) -> Dict:
        overall_start = time.time()

        max_workers = 4
        if self.config:
            max_workers = self.config.retrieval.faiss.max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # transform_vector in faiss_filter also runs FAISS-adjacent code —
            # hand it a CPU tensor so nothing downstream can call .numpy()
            # on a CUDA tensor.
            q_embed = self.faiss_retriever.transform_vector(
                question_embed.detach().cpu()
            )
            search_k = min(self.top_k * 3, 50)

            future_faiss_nodes = executor.submit(
                self._execute_faiss_node_search, self._as_cpu_numpy(q_embed), search_k
            )

            future_keywords = future_keyword_nodes = None
            if question:
                future_keywords = executor.submit(
                    self._extract_query_keywords, question
                )
                future_keyword_nodes = executor.submit(
                    self._get_keyword_based_nodes, future_keywords
                )

            future_faiss_relations = executor.submit(
                self._execute_faiss_relation_search, self._as_cpu_numpy(q_embed)
            )

            future_chunk_retrieval = executor.submit(
                self._chunk_embedding_retrieval, question_embed, self.top_k
            )

            faiss_candidate_nodes = future_faiss_nodes.result()

            future_faiss_sim = executor.submit(
                self._batch_calculate_entity_similarities,
                question_embed,
                faiss_candidate_nodes,
            )

            keyword_candidate_nodes = []
            if future_keyword_nodes:
                keyword_nodes = future_keyword_nodes.result()
                existing_faiss_nodes = set(faiss_candidate_nodes)
                keyword_candidate_nodes = [
                    n for n in keyword_nodes if n not in existing_faiss_nodes
                ]

            future_keyword_sim = (
                executor.submit(
                    self._batch_calculate_entity_similarities,
                    question_embed,
                    keyword_candidate_nodes,
                )
                if keyword_candidate_nodes
                else None
            )

            candidate_nodes = []
            faiss_similarities = future_faiss_sim.result()

            candidate_nodes.extend(
                (node, sim) for node, sim in faiss_similarities.items()
            )

            if future_keyword_sim:
                keyword_similarities = future_keyword_sim.result()

                candidate_nodes.extend(
                    (node, sim)
                    for node, sim in keyword_similarities.items()
                    if sim > 0.05
                )

            candidate_nodes.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [
                node for node, score in candidate_nodes[: self.top_k] if score > 0.05
            ]

            all_relations = future_faiss_relations.result()

            future_path_triples = (
                executor.submit(
                    self._path_based_search,
                    top_nodes,
                    future_keywords.result() if future_keywords else [],
                    max_depth=2,
                )
                if question
                else None
            )

            future_neighbor_triples = executor.submit(
                self._optimized_neighbor_expansion, top_nodes, question_embed
            )

            one_hop_triples = future_neighbor_triples.result()
            path_triples = future_path_triples.result() if future_path_triples else []
            relation_triples = self._get_relation_matched_triples(
                top_nodes, all_relations
            )

            all_triples = list(
                {triple for triple in one_hop_triples + path_triples + relation_triples}
            )
            chunk_results = future_chunk_retrieval.result()

        return {
            "top_nodes": top_nodes,
            "top_relations": all_relations,
            "one_hop_triples": all_triples,
            "chunk_results": chunk_results,
        }

    def _execute_faiss_node_search(self, q_embed, search_k: int) -> List[str]:
        """CUDA-safe FAISS node search."""
        q_np = self._as_cpu_numpy(q_embed).reshape(1, -1)
        _, I_nodes = self.faiss_retriever.node_index.search(q_np, search_k)
        return [
            self.faiss_retriever.node_map[str(idx)]
            for idx in I_nodes[0]
            if idx != -1 and str(idx) in self.faiss_retriever.node_map
        ]

    def _execute_faiss_relation_search(self, q_embed) -> List[str]:
        """CUDA-safe FAISS relation search."""
        q_np = self._as_cpu_numpy(q_embed).reshape(1, -1)
        _, I_relations = self.faiss_retriever.relation_index.search(q_np, self.top_k)
        return [
            self.faiss_retriever.relation_map[str(idx)]
            for idx in I_relations[0]
            if idx != -1 and str(idx) in self.faiss_retriever.relation_map
        ]

    def _get_keyword_based_nodes(self, future_keywords) -> List[str]:
        keywords = future_keywords.result()
        return self._keyword_based_node_search(keywords)

    @lru_cache(maxsize=1000)
    def _get_cached_neighbors(self, node_id: str) -> List[str]:
        return list(self.graph.neighbors(node_id))

    def _optimized_neighbor_expansion(
        self, top_nodes: List[str], question_embed: torch.Tensor
    ) -> List[Tuple]:
        all_neighbors = set()
        edge_queries = set()
        for node in top_nodes:
            neighbors = self._get_cached_neighbors(node)
            all_neighbors.update(neighbors)
            # Fix: use per-node `neighbors`, not the accumulated `all_neighbors`.
            edge_queries.update((node, n) for n in neighbors)
            edge_queries.update((n, node) for n in neighbors)

        triples = []
        for u, v in edge_queries:
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                relation = list(edge_data.values())[0].get("relation", "")
                if relation:
                    triples.append((u, relation, v))
        return triples

    def _get_relation_matched_triples(
        self, top_nodes: List[str], relations: List[str]
    ) -> List[Tuple]:
        top_node_set = set(top_nodes)
        relation_set = set(relations)

        return [
            (u, data.get("relation"), v)
            for u, v, data in self.graph.edges(data=True)
            if data.get("relation") in relation_set
            and (u in top_node_set or v in top_node_set)
        ]

    def _triple_only_retrieval(self, question_embed: torch.Tensor) -> Dict:
        """Path 2: Triple-only retrieval. CUDA-safe (hands FAISS a CPU tensor)."""
        try:
            # faiss_retriever.dual_path_retrieval internally calls .numpy() on
            # the query embedding. Detach + move to CPU here so it never sees
            # a CUDA tensor.
            qe_cpu = (
                question_embed.detach().cpu()
                if isinstance(question_embed, torch.Tensor)
                else question_embed
            )

            faiss_results = self.faiss_retriever.dual_path_retrieval(
                qe_cpu, top_k=self.top_k
            )

            scored_triples = faiss_results.get("scored_triples", [])

            return {"scored_triples": scored_triples}
        except Exception as e:
            logger.exception("Error in _triple_only_retrieval")
            return {"scored_triples": []}

    def _get_node_text(self, node: str) -> str:
        """Get text representation of a node."""
        if hasattr(self, "_node_text_cache") and node in self._node_text_cache:
            return self._node_text_cache[node]

        try:
            if node not in self.graph.nodes:
                return f"[Unknown Node: {node}]"

            data = self.graph.nodes[node]
            if "properties" in data and isinstance(data["properties"], dict):
                name = data["properties"].get("name", "")
                description = data["properties"].get("description", "")
            else:
                name = data.get("name", "")
                description = data.get("description", "")

            if isinstance(name, list):
                name = ", ".join(str(item) for item in name)
            elif not isinstance(name, str):
                name = str(name)

            if isinstance(description, list):
                description = ", ".join(str(item) for item in description)
            elif not isinstance(description, str):
                description = str(description)

            result = f"{name} {description}".strip()

            if not result or result.isspace():
                result = f"[Node: {node}]"

            if hasattr(self, "_node_text_cache"):
                self._node_text_cache[node] = result

            return result

        except Exception as e:
            logger.error(f"Error getting text for node {node}: {str(e)}")
            return f"[Error Node: {node}]"

    def _get_node_properties(self, node: str) -> str:
        """Get formatted properties of a node for display."""
        if node not in self.graph.nodes:
            return ""

        data = self.graph.nodes[node]
        properties = []

        SKIP_FIELDS = {
            "name",
            "description",
            "properties",
            "label",
            "chunk_id",
            "level",
            "class",
            "value", 
        }

        for source in [data.get("properties", {}), data]:
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if key in SKIP_FIELDS:
                    continue
                value_str = (
                    ", ".join(map(str, value))
                    if isinstance(value, list)
                    else str(value)
                )
                properties.append(f"{key}: {value_str}")

        return f"[{', '.join(properties)}]" if properties else ""

    def _extract_triple_based_info(
        self, triples: List[Tuple[str, str, str]]
    ) -> List[str]:
        """Extract readable information from triples with node properties."""
        triple_texts = []

        for h, r, t in triples:
            try:
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)
                head_props = self._get_node_properties(h)
                tail_props = self._get_node_properties(t)

                if (
                    head_text
                    and tail_text
                    and not head_text.startswith("[Error")
                    and not tail_text.startswith("[Error")
                ):
                    triple_text = (
                        f"({head_text} {head_props}, {r}, {tail_text} {tail_props})"
                    )
                    triple_texts.append(triple_text)
                else:
                    logger.info(f"Skipping triple with invalid nodes: ({h}, {r}, {t})")
            except Exception as e:
                logger.error(
                    f"Warning: Error processing triple ({h}, {r}, {t}): {str(e)}"
                )
                continue

        return triple_texts

    def _extract_scored_triple_info(
        self, scored_triples: List[Tuple[str, str, str, float]]
    ) -> List[str]:
        """Extract readable information from scored triples with node properties."""
        triples = []

        for i, (h, r, t, score) in enumerate(scored_triples):
            try:
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)
                head_props = self._get_node_properties(h)
                tail_props = self._get_node_properties(t)

                if (
                    head_text
                    and tail_text
                    and not head_text.startswith("[Error")
                    and not tail_text.startswith("[Error")
                ):
                    triple_text = (
                        f"({head_text} {head_props}, {r}, {tail_text} {tail_props}) "
                        f"[score: {score:.3f}]"
                    )
                    triples.append(triple_text)
                else:
                    logger.info(
                        f"Skipping scored triple with invalid nodes: ({h}, {r}, {t})"
                    )
            except Exception as e:
                logger.error(
                    f"Warning: Error processing scored triple ({h}, {r}, {t}): {str(e)}"
                )
                continue

        return triples

    def _parse_triple_string(self, triple: str) -> tuple[str, str, str, str]:
        """Parse a triple string and extract head, relation, tail, and score parts."""
        if not (triple.startswith("(") and triple.endswith(")")):
            return None, None, None, ""

        content = triple[1:-1]

        score_part = ""
        if " [score:" in content:
            content, score_suffix = content.split(" [score:", 1)
            score_part = f" [score:{score_suffix}"

        parts = self._split_respecting_brackets(content)

        if len(parts) < 3:
            return None, None, None, ""

        head = parts[0].strip()
        relation = parts[1].strip()
        tail = parts[2].strip()

        head_name = head.split(" [")[0] if " [" in head else head

        return head_name, relation, tail, score_part

    def _split_respecting_brackets(self, content: str) -> List[str]:
        """Split content by commas while respecting bracket nesting."""
        parts = []
        current_part = ""
        bracket_count = 0
        comma_count = 0

        for i, char in enumerate(content):
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            elif char == "," and bracket_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                comma_count += 1
                if comma_count == 2:
                    remaining = content[i + 1 :].strip()
                    if remaining:
                        parts.append(remaining)
                    break
                continue
            current_part += char

        if len(parts) < 3 and current_part.strip():
            parts.append(current_part.strip())

        return parts

    def _build_merged_triple(
        self, entity_name: str, relation: str, values: List[str]
    ) -> str:
        """Build a merged triple string from entity, relation, and values."""
        if len(values) == 1:
            return f"({entity_name}, {relation}, {values[0]})"
        else:
            merged_values = f"[{', '.join(values)}]"
            return f"({entity_name}, {relation}, {merged_values})"

    def _merge_entity_attributes(self, triples: List[str]) -> List[str]:
        """Merge multiple attributes of the same entity into a single list."""
        start_time = time.time()

        from collections import defaultdict

        entity_attributes = defaultdict(lambda: defaultdict(list))

        for triple in triples:
            try:
                head_name, relation, tail, score_part = self._parse_triple_string(
                    triple
                )

                if head_name and relation and tail is not None:
                    entity_attributes[head_name][relation].append(tail + score_part)

            except Exception as e:
                logger.error(f"Error processing triple {triple}: {str(e)}")
                continue

        merged_triples = [
            self._build_merged_triple(entity_name, relation, values)
            for entity_name, relations in entity_attributes.items()
            for relation, values in relations.items()
        ]

        elapsed = time.time() - start_time
        logger.info(f"[StepTiming] step=_merge_entity_attributes time={elapsed:.4f}")
        return merged_triples

    def _process_chunk_results(
        self, chunk_results: Dict, question_embed: torch.Tensor, top_k: int
    ) -> Tuple[List[str], set]:
        """Process chunk results and return formatted results and chunk IDs."""
        if not chunk_results:
            return [], set()

        reranked_results = self._rerank_chunks_by_relevance(
            chunk_results, question_embed, top_k
        )
        chunk_ids = reranked_results.get("chunk_ids", [])
        chunk_scores = reranked_results.get("scores", [])
        chunk_contents = reranked_results.get("chunk_contents", [])

        formatted_results = []
        chunk_id_set = set()

        for chunk_id, score, content in zip(chunk_ids, chunk_scores, chunk_contents):
            formatted_result = (
                f"[Chunk {chunk_id}] {content[:200]}... [score: {score:.3f}]"
            )
            formatted_results.append(formatted_result)
            chunk_id_set.add(chunk_id)

        return formatted_results, chunk_id_set

    def _collect_all_scored_triples(
        self, results: Dict, question_embed: torch.Tensor
    ) -> List[Tuple[str, str, str, float]]:
        """Collect and merge all scored triples from both paths."""
        all_scored_triples = []

        path2_scored = results.get("path2_results", {}).get("scored_triples", [])
        if path2_scored:
            all_scored_triples.extend(path2_scored)

        path1_triples = results.get("path1_results", {}).get("one_hop_triples", [])
        if path1_triples:
            path1_scored = self._rerank_triples_by_relevance(
                path1_triples, question_embed
            )
            all_scored_triples.extend(path1_scored)

        all_scored_triples.sort(key=lambda x: x[3], reverse=True)
        return all_scored_triples

    def _format_scored_triples(self, scored_triples):
        formatted_triples = []
        for h, r, t, score in scored_triples:
            head_text = self._get_node_text(h)
            tail_text = self._get_node_text(t)
            if (not head_text or not tail_text
                    or head_text.startswith("[Error")
                    or tail_text.startswith("[Error")):
                continue

            head_props = self._get_node_properties(h)
            tail_props = self._get_node_properties(t)

            # Edge-level provenance — this is the tight signal.
            edge_chunks = self._get_edge_chunk_ids(h, t, r)
            prov = f" [chunks: {', '.join(edge_chunks[:5])}" \
                f"{'…' if len(edge_chunks) > 5 else ''}]" if edge_chunks else ""

            if r in ("represented_by", "kw_filter_by"):
                continue

            formatted_triples.append(
                f"({head_text} {head_props}, {r}, {tail_text} {tail_props}) [score: {score:.3f}]"
            )
        return formatted_triples

    def _get_edge_chunk_ids(self, u: str, v: str, relation: str) -> List[str]:
        """Return chunk ids stored on all (u, v, relation) edges, deduped.

        A MultiDiGraph may hold multiple parallel edges between the same pair
        of nodes; kt_gen's `triple_deduplicate` normally collapses them to one,
        but pre-dedup JSONs or hand-edited graphs may not be.  Aggregating
        keeps the signal correct in either case.
        """
        if not self.graph.has_edge(u, v):
            return []

        edge_dict = self.graph.get_edge_data(u, v)  # {key: data} in MultiDiGraph
        if not edge_dict:
            return []

        # Ordered set: dict preserves insertion order on 3.7+; we only use the keys.
        seen: Dict[str, None] = {}
        for _key, data in edge_dict.items():
            if data.get("relation") != relation:
                continue
            for cid in data.get("chunk_id", []) or []:
                if cid:
                    seen.setdefault(str(cid), None)
        return list(seen)

    def _get_node_chunk_ids(self, node_data: dict) -> List[str]:
        """Return chunk ids recorded on a node, handling both layouts.

        kt_gen stores this under `properties.chunk_id` as a list of strings;
        some graphs also un-nest stringified lists like "['a', 'b']".  Returns
        a fresh list, never None, so callers can iterate unconditionally.
        """
        props = node_data.get("properties")
        raw = (
            props.get("chunk_id")
            if isinstance(props, dict)
            else node_data.get("chunk_id")
        )
        if not raw:
            return []
        if isinstance(raw, str):
            raw = [raw]

        out: List[str] = []
        for item in raw:
            try:
                parsed = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                parsed = item
            if isinstance(parsed, (list, tuple, set)):
                out.extend(str(c) for c in parsed if c)
            elif parsed:
                out.append(str(parsed))
        return out

    def _extract_chunk_ids_from_triples(
        self, scored_triples: List[Tuple[str, str, str, float]]
    ) -> set:
        """Extract chunk ids from scored triples, using EDGE provenance first.

        Edge provenance is the set of chunks in which this exact (head,
        relation, tail) transition happened.  Node provenance is used only as
        a fallback for triples whose edge somehow lacks provenance.
        """
        chunk_ids: set = set()

        for h, r, t, _score in scored_triples:
            edge_chunks = self._get_edge_chunk_ids(h, t, r)
            if edge_chunks:
                chunk_ids.update(edge_chunks)
                continue
            # Fallback: node-level provenance (older graphs, or edges that
            # were synthesised without provenance).
            for node in (h, t):
                if node in self.graph.nodes:
                    chunk_ids.update(self._get_node_chunk_ids(self.graph.nodes[node]))

        return chunk_ids

    def _get_matching_chunks(self, chunk_ids) -> Dict[str, str]:
        """Return {chunk_id: text} for ids that exist in chunk2id.

        Returning a dict (not a parallel list) makes downstream index
        alignment impossible to get wrong.
        """
        return {
            cid: self.chunk2id[cid]
            for cid in chunk_ids
            if cid in self.chunk2id
        }

    def process_retrieval_results(
        self, question: str, top_k: int = 20, involved_types: dict = None
    ) -> Tuple[Dict, float]:
        """Process retrieval results with edge-provenance as the primary chunk signal."""
        start_time = time.time()

        if involved_types:
            question_embed, results = self.retrieve_with_type_filtering(
                question, involved_types
            )
        else:
            question_embed, results = self.retrieve(question)

        retrieval_time = time.time() - start_time
        logger.info(f"retrieval time: {retrieval_time:.4f}")

        # ── Triples: rank and cap ────────────────────────────────────────────
        all_scored_triples = self._collect_all_scored_triples(results, question_embed)
        limited_scored_triples = all_scored_triples[:top_k]
        formatted_triples = self._format_scored_triples(limited_scored_triples)

        # ── Chunks: EDGE provenance, ranked by multi-triple support ──────────
        ranked_ids = self._rank_chunks_by_triple_support(limited_scored_triples, top_k)

        if len(ranked_ids) > top_k:
            selected_ids = self._semantic_rerank_within(
                ranked_ids, question_embed, top_k
            )
        else:
            selected_ids = ranked_ids

        # ── Fallback: dense retrieval only when provenance is empty ──────────
        if not selected_ids:
            logger.info(
                "No edge provenance produced chunks; falling back to dense retrieval."
            )
            selected_ids = self._dense_fallback(question_embed, top_k, results)

        matched = self._get_matching_chunks(selected_ids)
        final_ids      = list(matched.keys())           # iteration order preserved
        final_contents = [matched[cid] for cid in final_ids]

        retrieval_results = {
            "triples": formatted_triples,
            "chunk_ids": selected_ids,
            "chunk_contents": final_contents,
            "chunk_retrieval_results": [],  # deprecated field, kept for compat
        }

        return retrieval_results, retrieval_time

    def process_subquestions_parallel(
        self, sub_questions: List[Dict], top_k: int = 10, involved_types: dict = None
    ) -> Tuple[Dict, float]:
        """Process a list of sub-questions in parallel."""
        start_time = time.time()

        default_max_workers = 4
        if self.config:
            default_max_workers = self.config.retrieval.faiss.max_workers
        max_workers = min(len(sub_questions), default_max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_to_subquestion = {
                executor.submit(
                    self._process_single_subquestion, sub_q, top_k, involved_types
                ): sub_q
                for sub_q in sub_questions
            }
            all_triples = set()
            all_chunk_ids = set()
            all_chunk_contents = {}
            all_sub_question_results = []

            for future in concurrent.futures.as_completed(future_to_subquestion):
                sub_q = future_to_subquestion[future]
                try:
                    sub_result = future.result()

                    with threading.Lock():
                        all_triples.update(sub_result["triples"])
                        all_chunk_ids.update(sub_result["chunk_ids"])

                        for chunk_id, content in sub_result["chunk_contents"].items():
                            all_chunk_contents[chunk_id] = content

                        all_sub_question_results.append(sub_result["sub_result"])
                except Exception as e:
                    logger.error(f"Error processing sub-question: {str(e)}")
                    with threading.Lock():
                        all_sub_question_results.append(
                            {
                                "sub_question": sub_q.get("sub-question", ""),
                                "triples_count": 0,
                                "chunk_ids_count": 0,
                                "time_taken": 0.0,
                            }
                        )

        dedup_triples = list(all_triples)
        dedup_chunk_ids = list(all_chunk_ids)

        dedup_chunk_contents = {
            chunk_id: all_chunk_contents.get(
                chunk_id, f"[Missing content for chunk {chunk_id}]"
            )
            for chunk_id in dedup_chunk_ids
        }

        if not dedup_triples and not dedup_chunk_contents:
            dedup_triples = ["No relevant information found"]
            dedup_chunk_contents = {"no_chunks": "No relevant chunks found"}

        total_time = time.time() - start_time

        return {
            "triples": dedup_triples,
            "chunk_ids": dedup_chunk_ids,
            "chunk_contents": dedup_chunk_contents,
            "sub_question_results": all_sub_question_results,
        }, total_time

    def _process_single_subquestion(
        self, sub_question: Dict, top_k: int, involved_types: dict = None
    ) -> Dict:
        sub_question_text = sub_question.get("sub-question", "")
        try:
            retrieval_results, time_taken = self.process_retrieval_results(
                sub_question_text, top_k, involved_types
            )
            triples = retrieval_results.get("triples", []) or []
            chunk_ids = retrieval_results.get("chunk_ids", []) or []
            chunk_contents = retrieval_results.get("chunk_contents", []) or []

            if isinstance(chunk_contents, dict):
                chunk_contents_list = list(chunk_contents.values())
            else:
                chunk_contents_list = chunk_contents

            if not isinstance(triples, (list, tuple)):
                logger.warning(f"triples is not a list: {type(triples)}")
                triples = []
            if not isinstance(chunk_ids, (list, tuple)):
                logger.warning(f"chunk_ids is not a list: {type(chunk_ids)}")
                chunk_ids = []
            if not isinstance(chunk_contents_list, (list, tuple)):
                logger.warning(
                    f"chunk_contents_list is not a list: {type(chunk_contents_list)}"
                )
                chunk_contents_list = []

            sub_result = {
                "sub_question": sub_question_text,
                "triples_count": len(triples),
                "chunk_ids_count": len(chunk_ids),
                "time_taken": time_taken,
            }

            chunk_contents_dict = {}
            for i, chunk_id in enumerate(chunk_ids):
                if i < len(chunk_contents_list):
                    chunk_contents_dict[chunk_id] = chunk_contents_list[i]
                else:
                    chunk_contents_dict[chunk_id] = (
                        f"[Missing content for chunk {chunk_id}]"
                    )

            return {
                "triples": set(triples),
                "chunk_ids": set(chunk_ids),
                "chunk_contents": chunk_contents_dict,
                "sub_result": sub_result,
            }

        except Exception as e:
            logger.error(
                f"Error processing sub-question '{sub_question_text}': {str(e)}"
            )
            return {
                "triples": set(),
                "chunk_ids": set(),
                "chunk_contents": {},
                "sub_result": {
                    "sub_question": sub_question_text,
                    "triples_count": 0,
                    "chunk_ids_count": 0,
                    "time_taken": 0.0,
                },
            }

    def generate_answer(self, prompt: str) -> str:
        answer = self.llm_client.call_api(prompt)
        logger.info("Retrieved context:")
        logger.info(prompt)
        logger.info("Final Answer:")
        logger.info(answer)
        return answer

    def _extract_chunk_ids_from_nodes(self, nodes: List[str]) -> List[str]:
        flat: Dict[str, None] = {}
        for node in nodes:
            if node not in self.graph.nodes:
                continue
            for cid in self._get_node_chunk_ids(self.graph.nodes[node]):
                flat.setdefault(cid, None)
        return sorted(flat)

    def _enhance_query_with_entities(self, question: str) -> str:
        """Enhance query by extracting entities and relations using spaCy NER."""
        try:
            doc = self.nlp(question)

            entities = []
            for ent in doc.ents:
                entities.append(ent.text)

            key_phrases = []
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop:
                    key_phrases.append(token.text)
                    if len(key_phrases) >= 5:
                        break

            enhanced_parts = [question]
            if entities:
                enhanced_parts.append(f"Entities: {', '.join(entities)}")
            if key_phrases:
                enhanced_parts.append(f"Key terms: {', '.join(key_phrases)}")

            enhanced_query = " ".join(enhanced_parts)

            return enhanced_query

        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return question

    def _calculate_entity_similarity(
        self, query_embed: torch.Tensor, node: str
    ) -> float:
        """Calculate entity-level similarity between query and node."""
        try:
            if node not in self.graph.nodes:
                return 0.0

            node_text = self._get_node_text(node)
            if (
                not node_text
                or node_text.startswith("[Error")
                or node_text.startswith("[Unknown")
            ):
                return 0.0

            if node in self.node_embedding_cache:
                node_embed = self.node_embedding_cache[node]
            else:
                node_embed = (
                    torch.tensor(self.qa_encoder.encode(node_text))
                    .float()
                    .to(self.device)
                )
                self.node_embedding_cache[node] = node_embed

            similarity = F.cosine_similarity(query_embed, node_embed, dim=0).item()
            similarity = max(0.0, similarity)

            return similarity

        except Exception as e:
            logger.error(
                f"Error calculating entity similarity for node {node}: {str(e)}"
            )
            return 0.0

    def _batch_calculate_entity_similarities(
        self, query_embed: torch.Tensor, nodes: List[str]
    ) -> Dict[str, float]:
        """Batch entity similarity. Runs on self.device — the query_embed must
        be on the same device as the cached node embeddings."""
        similarities = {}
        node_embeddings = []
        valid_nodes = []
        with self.cache_locks["node_embedding"]:
            for node in nodes:
                if node in self.node_embedding_cache:
                    node_embeddings.append(self.node_embedding_cache[node])
                    valid_nodes.append(node)

        if node_embeddings:
            try:
                node_embeddings_tensor = torch.stack(node_embeddings)

                batch_similarities = F.cosine_similarity(
                    query_embed.unsqueeze(0), node_embeddings_tensor, dim=1
                )

                for i, node in enumerate(valid_nodes):
                    similarity = max(0.0, batch_similarities[i].item())
                    similarities[node] = similarity

            except Exception:
                for node in valid_nodes:
                    try:
                        similarity = self._calculate_entity_similarity(
                            query_embed, node
                        )
                        similarities[node] = similarity
                    except Exception as e2:
                        logger.error(
                            f"Error calculating similarity for node {node}: {str(e2)}"
                        )
                        continue
        else:
            for node in nodes:
                try:
                    similarity = self._calculate_entity_similarity(query_embed, node)
                    similarities[node] = similarity
                except Exception as e:
                    logger.error(
                        f"Error calculating similarity for node {node}: {str(e)}"
                    )
                    continue

        return similarities

    def _smart_neighbor_expansion(
        self, center_node: str, query_embed: torch.Tensor, max_neighbors: int = 5
    ) -> List[str]:
        """Optimized smart neighbor expansion with batch similarity calculation."""
        if center_node not in self.graph.nodes:
            return []

        neighbors = list(self.graph.neighbors(center_node))
        if not neighbors:
            return []

        valid_neighbors = [n for n in neighbors if n in self.graph.nodes]
        if not valid_neighbors:
            return []

        neighbor_similarities = self._batch_calculate_entity_similarities(
            query_embed, valid_neighbors
        )

        sorted_neighbors = sorted(
            neighbor_similarities.items(), key=lambda x: x[1], reverse=True
        )

        return [node for node, score in sorted_neighbors[:max_neighbors] if score > 0.1]

    def _rerank_triples_by_relevance(
        self, triples: List[Tuple[str, str, str]], question_embed: torch.Tensor
    ) -> List[Tuple[str, str, str, float]]:
        """Optimized triple reranking with batch encoding."""
        start_time = time.time()
        if not triples:
            return []

        scored_triples = []
        triple_texts = []
        valid_triples = []

        for h, r, t in triples:
            try:
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)

                if (
                    not head_text
                    or not tail_text
                    or head_text.startswith("[Error")
                    or tail_text.startswith("[Error")
                ):
                    continue

                triple_text = f"{head_text} {r} {tail_text}"
                triple_texts.append(triple_text)
                valid_triples.append((h, r, t))

            except Exception as e:
                logger.error(f"Error processing triple ({h}, {r}, {t}): {str(e)}")
                continue

        if not valid_triples:
            return []

        try:
            encode_start = time.time()
            triple_embeddings = self.qa_encoder.encode(
                triple_texts, convert_to_tensor=True
            ).to(self.device)
            encode_elapsed = time.time() - encode_start
            logger.info(
                f"[StepTiming] step=batch_encode_triple_texts time={encode_elapsed:.4f}"
            )

            sim_calc_start = time.time()
            similarities = F.cosine_similarity(
                question_embed.unsqueeze(0), triple_embeddings, dim=1
            )
            sim_calc_elapsed = time.time() - sim_calc_start
            logger.info(
                f"[StepTiming] step=batch_calculate_similarities time={sim_calc_elapsed:.4f}"
            )

            for i, (h, r, t) in enumerate(valid_triples):
                similarity = similarities[i].item()

                relation_bonus = 0.0
                if r.lower() in [
                    "is",
                    "was",
                    "has",
                    "had",
                    "contains",
                    "located",
                    "born",
                    "died",
                ]:
                    relation_bonus = 0.1

                final_score = max(0.0, similarity + relation_bonus)

                if final_score > 0.05:
                    scored_triples.append((h, r, t, final_score))

        except Exception as e:
            logger.error(f"Error in batch triple encoding: {str(e)}")
            return self._rerank_triples_individual(triples, question_embed)

        scored_triples.sort(key=lambda x: x[3], reverse=True)
        elapsed = time.time() - start_time
        logger.info(
            f"[StepTiming] step=_rerank_triples_by_relevance time={elapsed:.4f}"
        )
        return scored_triples

    def _rerank_triples_individual(
        self, triples: List[Tuple[str, str, str]], question_embed: torch.Tensor
    ) -> List[Tuple[str, str, str, float]]:
        """Fallback individual triple processing."""
        scored_triples = []

        for h, r, t in triples:
            try:
                head_text = self._get_node_text(h)
                tail_text = self._get_node_text(t)

                if (
                    not head_text
                    or not tail_text
                    or head_text.startswith("[Error")
                    or tail_text.startswith("[Error")
                ):
                    continue

                triple_text = f"{head_text} {r} {tail_text}"
                triple_embed = (
                    torch.tensor(self.qa_encoder.encode(triple_text))
                    .float()
                    .to(self.device)
                )
                similarity = F.cosine_similarity(
                    question_embed, triple_embed, dim=0
                ).item()
                relation_bonus = 0.0
                if r.lower() in [
                    "is",
                    "was",
                    "has",
                    "had",
                    "contains",
                    "located",
                    "born",
                    "died",
                ]:
                    relation_bonus = 0.1

                final_score = max(0.0, similarity + relation_bonus)

                if final_score > 0.05:
                    scored_triples.append((h, r, t, final_score))

            except Exception as e:
                logger.error(f"Error reranking triple ({h}, {r}, {t}): {str(e)}")
                continue

        scored_triples.sort(key=lambda x: x[3], reverse=True)
        return scored_triples

    def _extract_query_keywords(self, question: str) -> List[str]:
        """Automatically extract keywords from the question using spaCy."""
        try:
            doc = self.nlp(question.lower())
            keywords = []

            for token in doc:
                if not token.is_stop and len(token.text) > 2:
                    if token.ent_type_:
                        keywords.append(token.text.lower())
                    elif token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                        keywords.append(token.text.lower())
                    elif token.pos_ == "VERB":
                        keywords.append(token.text.lower())

            for ent in doc.ents:
                if len(ent.text) > 2:
                    keywords.append(ent.text.lower())

            unique_keywords = list(set(keywords))
            return unique_keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def _keyword_based_node_search(self, keywords: List[str]) -> List[str]:
        """Optimized keyword-based node search."""
        if not keywords:
            return []

        use_exact_matching = getattr(self, "use_exact_keyword_matching", True)

        if use_exact_matching:
            if not hasattr(self, "_node_text_index") or self._node_text_index is None:
                logger.warning(
                    "Node text index not found. This should be built during initialization."
                )
                return []

            relevant_nodes = set()
            max_nodes_per_keyword = 50

            for keyword in keywords:
                if keyword in self._node_text_index:
                    keyword_nodes = self._node_text_index[keyword]
                    if len(keyword_nodes) > max_nodes_per_keyword:
                        keyword_nodes = set(list(keyword_nodes)[:max_nodes_per_keyword])
                    relevant_nodes.update(keyword_nodes)
                else:
                    continue

                if len(relevant_nodes) > 200:
                    break
            return list(relevant_nodes)
        else:
            return self._keyword_based_node_search_original(keywords)

    def _keyword_based_node_search_original(self, keywords: List[str]) -> List[str]:
        """Original keyword-based node search with substring matching."""
        relevant_nodes = []
        for node in self.graph.nodes():
            try:
                node_text = self._get_node_text(node).lower()

                for keyword in keywords:
                    if keyword in node_text:
                        relevant_nodes.append(node)
                        break

            except Exception:
                continue

        return relevant_nodes

    def _build_node_text_index(self):
        """Build inverted index for node texts to speed up keyword search."""
        if self._load_node_text_index():
            logger.info("Loaded node text index from cache")
            return

        start_time = time.time()
        logger.info("Building optimized node text index for keyword search...")
        self._node_text_index = {}

        if hasattr(self, "_node_text_cache") and self._node_text_cache:
            node_texts = self._node_text_cache
        else:
            node_texts = {}
            for node in self.graph.nodes():
                node_texts[node] = self._get_node_text(node)

        total_nodes = len(node_texts)
        processed_nodes = 0

        for node, node_text in node_texts.items():
            try:
                node_text_lower = node_text.lower()
                words = set(node_text_lower.split())

                for word in words:
                    if len(word) > 2:
                        if word not in self._node_text_index:
                            self._node_text_index[word] = set()
                        self._node_text_index[word].add(node)

                processed_nodes += 1
                if processed_nodes % 1000 == 0:
                    logger.info(f"Indexed {processed_nodes}/{total_nodes} nodes")

            except Exception as e:
                logger.error(f"Error indexing node {node}: {str(e)}")
                continue

        end_time = time.time()
        logger.info(
            f"Time taken to build node text index: {end_time - start_time} seconds"
        )

        self._save_node_text_index()

    def _save_node_text_index(self):
        """Save node text index to disk cache."""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_index.pkl"
        try:
            if not self._node_text_index:
                logger.warning("No node text index to save!")
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            serializable_index = {}
            for word, nodes in self._node_text_index.items():
                serializable_index[word] = list(nodes)

            with open(cache_path, "wb") as f:
                pickle.dump(serializable_index, f)

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"Saved node text index with {len(serializable_index)} words to "
                f"{cache_path} (size: {file_size} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving node text index: {e}")
            return False

    def _load_node_text_index(self):
        """Load node text index from disk cache."""
        cache_path = f"{self.cache_dir}/{self.dataset}/node_text_index.pkl"
        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    logger.warning(
                        f"Cache file too small ({file_size} bytes), likely empty or corrupted"
                    )
                    return False

                with open(cache_path, "rb") as f:
                    serializable_index = pickle.load(f)

                if not serializable_index:
                    logger.warning("Loaded index is empty")
                    return False

                self._node_text_index = {}
                for word, nodes in serializable_index.items():
                    self._node_text_index[word] = set(nodes)

                if not self._check_text_index_consistency():
                    logger.info(
                        "Text index inconsistent with current graph, will rebuild"
                    )
                    return False

                logger.info(
                    f"Loaded node text index with {len(self._node_text_index)} words from "
                    f"{cache_path} (file size: {file_size} bytes)"
                )
                return True

            except Exception as e:
                logger.error(f"Error loading node text index: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
                except Exception as e2:
                    logger.error(
                        f"Failed to remove corrupted cache file {cache_path}: {type(e2).__name__}: {e2}"
                    )
        else:
            logger.info(f"Cache file not found: {cache_path}")
        return False

    def _check_text_index_consistency(self):
        """Check if the loaded text index is consistent with current graph."""
        try:
            indexed_nodes = set()
            for nodes in self._node_text_index.values():
                indexed_nodes.update(nodes)

            current_nodes = set(self.graph.nodes())
            missing_nodes = current_nodes - indexed_nodes
            if missing_nodes:
                logger.warning(
                    f"Text index missing {len(missing_nodes)} nodes from current graph"
                )
                return False

            extra_nodes = indexed_nodes - current_nodes
            if len(extra_nodes) > len(current_nodes) * 0.1:
                logger.warning(
                    f"Text index has too many extra nodes: {len(extra_nodes)} extra "
                    f"vs {len(current_nodes)} current"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking text index consistency: {e}")
            return False

    def _path_based_search(
        self, start_nodes: List[str], target_keywords: List[str], max_depth: int = 2
    ) -> List[Tuple[str, str, str]]:
        """Search for paths from start nodes to nodes containing target keywords."""
        found_triples = []
        visited = set()

        def dfs_search(node: str, depth: int, path: List[str]):
            if depth > max_depth or node in visited:
                return

            visited.add(node)

            try:
                node_text = self._get_node_text(node).lower()
                for keyword in target_keywords:
                    if keyword in node_text:
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_data = self.graph.get_edge_data(u, v)
                            if edge_data and "relation" in edge_data:
                                relation = list(edge_data.values())[0]["relation"]
                                found_triples.append((u, relation, v))
                        break
            except Exception as e:
                logger.warning(
                    f"Error during DFS path search at node "
                    f"{start_node if 'start_node' in locals() else ''}: "
                    f"{type(e).__name__}: {e}"
                )

            if depth < max_depth:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        dfs_search(neighbor, depth + 1, path + [neighbor])

        for start_node in start_nodes:
            dfs_search(start_node, 0, [start_node])

        return found_triples

    def _precompute_chunk_embeddings(self):
        """Precompute embeddings for all chunks to enable direct chunk retrieval."""
        with self.precompute_lock:
            if self.chunk_embeddings_precomputed:
                return

            logger.info("Precomputing chunk embeddings for direct chunk retrieval...")
            if self._load_chunk_embedding_cache():
                logger.info("Successfully loaded chunk embeddings from disk cache")
                self.chunk_embeddings_precomputed = True
                return

            if not self.chunk2id:
                logger.info("Warning: No chunks available for embedding computation")
                return

            logger.info("Computing chunk embeddings from scratch...")

            chunk_ids = list(self.chunk2id.keys())
            chunk_texts = list(self.chunk2id.values())
            batch_size = 50
            if self.config:
                batch_size = self.config.embeddings.batch_size

            total_processed = 0
            embeddings_list = []
            valid_chunk_ids = []

            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i : i + batch_size]
                batch_chunk_ids = chunk_ids[i : i + batch_size]

                try:
                    batch_embeddings = self.qa_encoder.encode(
                        batch_texts, convert_to_tensor=True
                    )

                    for j, chunk_id in enumerate(batch_chunk_ids):
                        self.chunk_embedding_cache[chunk_id] = batch_embeddings[j]
                        embeddings_list.append(
                            batch_embeddings[j].detach().cpu().numpy()
                        )
                        valid_chunk_ids.append(chunk_id)
                        total_processed += 1

                except Exception as e:
                    logger.error(
                        f"Error encoding chunk batch {i // batch_size}: {str(e)}"
                    )
                    for j, chunk_id in enumerate(batch_chunk_ids):
                        try:
                            chunk_text = self.chunk2id[chunk_id]
                            embedding = (
                                torch.tensor(self.qa_encoder.encode(chunk_text))
                                .float()
                                .to(self.device)
                            )
                            self.chunk_embedding_cache[chunk_id] = embedding
                            embeddings_list.append(embedding.detach().cpu().numpy())
                            valid_chunk_ids.append(chunk_id)
                            total_processed += 1
                        except Exception as e2:
                            logger.error(f"Error encoding chunk {chunk_id}: {str(e2)}")
                            continue

            if embeddings_list:
                try:
                    logger.info("Building FAISS index for chunk embeddings...")
                    embeddings_array = np.array(embeddings_list, dtype="float32")
                    dimension = embeddings_array.shape[1]

                    self.chunk_faiss_index = faiss.IndexFlatIP(dimension)
                    self.chunk_faiss_index.add(embeddings_array)

                    for i, chunk_id in enumerate(valid_chunk_ids):
                        self.chunk_id_to_index[chunk_id] = i
                        self.index_to_chunk_id[i] = chunk_id

                    logger.info(f"FAISS index built with {len(valid_chunk_ids)} chunks")

                except Exception as e:
                    logger.error(f"Error building FAISS index for chunks: {str(e)}")

            self.chunk_embeddings_precomputed = True
            logger.info(
                f"Chunk embeddings precomputed for {total_processed} chunks "
                f"(cache size: {len(self.chunk_embedding_cache)})"
            )

            self._save_chunk_embedding_cache()

    def _save_chunk_embedding_cache(self):
        """Save chunk embedding cache to disk."""
        cache_path = f"{self.cache_dir}/{self.dataset}/chunk_embedding_cache.pt"
        try:
            if not self.chunk_embedding_cache:
                return False

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            numpy_cache = {}
            for chunk_id, embed in self.chunk_embedding_cache.items():
                if embed is not None:
                    try:
                        if hasattr(embed, "detach"):
                            numpy_cache[chunk_id] = embed.detach().cpu().numpy()
                        elif isinstance(embed, np.ndarray):
                            numpy_cache[chunk_id] = embed
                        else:
                            numpy_cache[chunk_id] = np.array(embed)
                    except Exception:
                        continue

            if not numpy_cache:
                return False

            try:
                tensor_cache = {}
                for chunk_id, embed_array in numpy_cache.items():
                    if isinstance(embed_array, np.ndarray):
                        tensor_cache[chunk_id] = torch.from_numpy(embed_array).float()
                    else:
                        tensor_cache[chunk_id] = embed_array

                torch.save(tensor_cache, cache_path)
            except Exception:
                cache_path_npz = cache_path.replace(".pt", ".npz")
                np.savez_compressed(cache_path_npz, **numpy_cache)
                cache_path = cache_path_npz

            file_size = os.path.getsize(cache_path)
            logger.info(
                f"Saved chunk embedding cache with {len(numpy_cache)} entries to "
                f"{cache_path} (size: {file_size} bytes)"
            )
            return True

        except Exception:
            return False

    def _load_chunk_embedding_cache(self):
        """Load chunk embedding cache from disk."""
        cache_path = f"{self.cache_dir}/{self.dataset}/chunk_embedding_cache.pt"
        cache_path_npz = cache_path.replace(".pt", ".npz")

        if os.path.exists(cache_path_npz):
            try:
                file_size = os.path.getsize(cache_path_npz)
                numpy_cache = np.load(cache_path_npz)

                if len(numpy_cache.files) == 0:
                    return False

                self.chunk_embedding_cache.clear()

                for chunk_id in numpy_cache.files:
                    try:
                        embed_array = numpy_cache[chunk_id]
                        embed_tensor = (
                            torch.from_numpy(embed_array).float().to(self.device)
                        )
                        self.chunk_embedding_cache[chunk_id] = embed_tensor
                    except Exception:
                        continue

                numpy_cache.close()

                logger.info(
                    f"Loaded chunk embedding cache with {len(self.chunk_embedding_cache)} "
                    f"entries from {cache_path_npz}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to load chunk embedding cache from {cache_path_npz}: {e}"
                )
                return False

        if os.path.exists(cache_path):
            try:
                file_size = os.path.getsize(cache_path)
                if file_size < 1000:
                    return False

                try:
                    cpu_cache = torch.load(
                        cache_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    cpu_cache = torch.load(cache_path, map_location="cpu")
                except Exception as e:
                    if "numpy.core.multiarray._reconstruct" in str(e):
                        try:
                            import importlib

                            torch_serialization = importlib.import_module(
                                "torch.serialization"
                            )
                            torch_serialization.add_safe_globals(
                                ["numpy.core.multiarray._reconstruct"]
                            )
                            cpu_cache = torch.load(cache_path, map_location="cpu")
                        except Exception:
                            raise e
                    else:
                        raise e

                if not cpu_cache:
                    logger.warning(f"Chunk embedding cache is empty from {cache_path}")
                    return False

                self.chunk_embedding_cache.clear()

                for chunk_id, embed in cpu_cache.items():
                    if embed is not None:
                        try:
                            if isinstance(embed, np.ndarray):
                                embed_tensor = torch.from_numpy(embed).float()
                            else:
                                embed_tensor = (
                                    embed.cpu() if hasattr(embed, "cpu") else embed
                                )

                            if self.device == "cuda" and torch.cuda.is_available():
                                embed_tensor = embed_tensor.to(self.device)
                            else:
                                embed_tensor = embed_tensor.to("cpu")

                            self.chunk_embedding_cache[chunk_id] = embed_tensor
                        except Exception as e:
                            logger.error(
                                f"Warning: Failed to load chunk embedding for {chunk_id}: {e}"
                            )
                            continue

                if self.chunk_embedding_cache:
                    try:
                        embeddings_list = []
                        valid_chunk_ids = []

                        for chunk_id, embed in self.chunk_embedding_cache.items():
                            embeddings_list.append(embed.detach().cpu().numpy())
                            valid_chunk_ids.append(chunk_id)

                        embeddings_array = np.array(embeddings_list, dtype="float32")
                        dimension = embeddings_array.shape[1]

                        self.chunk_faiss_index = faiss.IndexFlatIP(dimension)
                        self.chunk_faiss_index.add(embeddings_array)

                        self.chunk_id_to_index.clear()
                        self.index_to_chunk_id.clear()
                        for i, chunk_id in enumerate(valid_chunk_ids):
                            self.chunk_id_to_index[chunk_id] = i
                            self.index_to_chunk_id[i] = chunk_id

                    except Exception:
                        return False

                if not self._check_chunk_cache_consistency():
                    return False

                logger.info(
                    f"Loaded chunk embedding cache with {len(self.chunk_embedding_cache)} "
                    f"entries from {cache_path} (file size: {file_size} bytes)"
                )
                return True

            except Exception as e:
                logger.error(f"Error loading chunk embedding cache: {e}")
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted chunk cache file: {cache_path}")
                except Exception as e:
                    logger.error(
                        f"Error removing corrupted chunk cache file: {cache_path}: {e}"
                    )
        else:
            logger.info(f"Chunk cache file not found: {cache_path}")
        return False

    def _check_chunk_cache_consistency(self):
        """Check if the loaded chunk cache is consistent with current chunks."""
        try:
            current_chunk_ids = set(self.chunk2id.keys())
            cached_chunk_ids = set(self.chunk_embedding_cache.keys())

            missing_chunks = current_chunk_ids - cached_chunk_ids
            if missing_chunks:
                logger.info(
                    f"Chunk cache missing {len(missing_chunks)} chunks from current chunks"
                )
                return False

            extra_chunks = cached_chunk_ids - current_chunk_ids
            if len(extra_chunks) > len(current_chunk_ids) * 0.1:
                logger.info(
                    f"Chunk cache has too many extra chunks: {len(extra_chunks)} extra "
                    f"vs {len(current_chunk_ids)} current"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking chunk cache consistency: {e}")
            return False

    def _chunk_embedding_retrieval(
        self, question_embed: torch.Tensor, top_k: int = 20
    ) -> Dict:
        """CUDA-safe FAISS chunk retrieval."""
        try:
            if not self.chunk_embeddings_precomputed or self.chunk_faiss_index is None:
                logger.info(
                    "Warning: Chunk embeddings not precomputed, skipping chunk retrieval"
                )
                return {"chunk_ids": [], "scores": [], "chunk_contents": []}

            # === CUDA-safe: use the helper (detach + cpu + float32 + contiguous) ===
            query_embed_np = self._as_cpu_numpy(question_embed).reshape(1, -1)
            scores, indices = self.chunk_faiss_index.search(
                query_embed_np, min(top_k, self.chunk_faiss_index.ntotal)
            )

            chunk_ids = []
            similarity_scores = []
            chunk_contents = []

            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx in self.index_to_chunk_id:
                    chunk_id = self.index_to_chunk_id[idx]
                    chunk_ids.append(chunk_id)
                    similarity_scores.append(float(score))

                    if chunk_id in self.chunk2id:
                        chunk_contents.append(self.chunk2id[chunk_id])
                    else:
                        chunk_contents.append(f"[Missing content for chunk {chunk_id}]")

            return {
                "chunk_ids": chunk_ids,
                "scores": similarity_scores,
                "chunk_contents": chunk_contents,
            }

        except Exception as e:
            logger.exception("Error in chunk embedding retrieval")
            return {"chunk_ids": [], "scores": [], "chunk_contents": []}

    def _rank_chunks_by_triple_support(
        self,
        scored_triples: List[Tuple[str, str, str, float]],
        top_k: int,
    ) -> List[str]:
        """Rank chunk ids by how many top-k triples cite them (weighted by score).

        A chunk cited by three high-scoring triples is preferred over a chunk
        cited by one.  Ties broken by first-seen order (stable ordering).

        Uses EDGE provenance.  This is the primary chunk signal for retrieval.
        """
        from collections import defaultdict

        weighted: Dict[str, float] = defaultdict(float)
        support: Dict[str, int] = defaultdict(int)
        order: Dict[str, int] = {}

        for h, r, t, score in scored_triples:
            edge_chunks = self._get_edge_chunk_ids(h, t, r)
            if not edge_chunks:
                # No edge provenance — fall back to node provenance for this
                # triple only, with a penalty so genuine edge hits outrank it.
                edge_chunks = []
                for node in (h, t):
                    if node in self.graph.nodes:
                        edge_chunks.extend(
                            self._get_node_chunk_ids(self.graph.nodes[node])
                        )

            for cid in edge_chunks:
                if cid not in order:
                    order[cid] = len(order)
                weighted[cid] += score
                support[cid] += 1

        ranked = sorted(
            weighted.keys(),
            key=lambda c: (-support[c], -weighted[c], order[c]),
        )
        return ranked[:top_k]

    def _semantic_rerank_within(
        self,
        chunk_ids: List[str],
        question_embed: torch.Tensor,
        top_k: int,
    ) -> List[str]:
        """Rerank an existing candidate chunk-id list by cosine similarity.

        Encodes ONLY the candidate chunks — not the whole corpus.  This is
        cheap (dozens of chunks, not thousands) and only fires when the
        provenance set is larger than `top_k`.
        """
        valid = [cid for cid in chunk_ids if cid in self.chunk2id]
        if not valid:
            return chunk_ids[:top_k]

        texts = [self.chunk2id[cid] for cid in valid]
        try:
            embeds = self.qa_encoder.encode(texts, convert_to_tensor=True).to(
                self.device
            )
            sims = F.cosine_similarity(question_embed.unsqueeze(0), embeds, dim=1)
            order = sorted(range(len(valid)), key=lambda i: -sims[i].item())
            return [valid[i] for i in order[:top_k]]
        except Exception as e:
            logger.warning(f"Semantic prune failed, using score order: {e}")
            return valid[:top_k]

    def _dense_fallback(
        self,
        question_embed: torch.Tensor,
        top_k: int,
        results: Dict,
    ) -> List[str]:
        """Fallback chunk signal when no triple produced edge provenance.

        Preference order:
        1. path1_results.chunk_results (present in non-type-filtered mode)
        2. Direct _chunk_embedding_retrieval (works in every mode)
        """
        chunk_results = results.get("path1_results", {}).get("chunk_results")
        if chunk_results:
            _, dense_ids = self._process_chunk_results(
                chunk_results, question_embed, top_k
            )
            return list(dense_ids)

        dense = self._chunk_embedding_retrieval(question_embed, top_k)
        return list(dense.get("chunk_ids", []))

    def _rerank_chunks_by_relevance(
        self, chunk_results: Dict, question_embed: torch.Tensor, top_k: int = 10
    ) -> Dict:
        """Rerank chunks by relevance to the question using semantic similarity."""
        try:
            chunk_ids = chunk_results.get("chunk_ids", [])
            original_scores = chunk_results.get("scores", [])
            chunk_contents = chunk_results.get("chunk_contents", [])

            if not chunk_ids or not chunk_contents:
                return chunk_results

            chunk_similarities = []
            for i, (chunk_id, content) in enumerate(zip(chunk_ids, chunk_contents)):
                try:
                    chunk_embed = (
                        torch.tensor(self.qa_encoder.encode(content))
                        .float()
                        .to(self.device)
                    )

                    similarity = F.cosine_similarity(
                        question_embed, chunk_embed, dim=0
                    ).item()
                    similarity = max(0.0, similarity)

                    faiss_score = (
                        original_scores[i] if i < len(original_scores) else 0.0
                    )
                    combined_score = (faiss_score + similarity) / 2.0

                    chunk_similarities.append((chunk_id, content, combined_score, i))

                except Exception as e:
                    logger.error(
                        f"Error calculating similarity for chunk {chunk_id}: {str(e)}"
                    )
                    faiss_score = (
                        original_scores[i] if i < len(original_scores) else 0.0
                    )
                    chunk_similarities.append((chunk_id, content, faiss_score, i))

            chunk_similarities.sort(key=lambda x: x[2], reverse=True)

            top_chunks = chunk_similarities[:top_k]

            reranked_chunk_ids = [chunk_id for chunk_id, _, _, _ in top_chunks]
            reranked_scores = [score for _, _, score, _ in top_chunks]
            reranked_contents = [content for _, content, _, _ in top_chunks]

            return {
                "chunk_ids": reranked_chunk_ids,
                "scores": reranked_scores,
                "chunk_contents": reranked_contents,
            }

        except Exception as e:
            logger.error(f"Error in chunk reranking: {str(e)}")
            return chunk_results
