# models/retriever/agentic_decomposer.py
import json_repair
from utils import call_llm_api

try:
    from config import get_config
except ImportError:
    get_config = None


class GraphQ:
    def __init__(self, dataset_name, config=None):
        if config is None and get_config is not None:
            try:
                self.config = get_config()
            except Exception:
                self.config = None
        else:
            self.config = config
        self.llm_client = call_llm_api.LLMCompletionCall()
        self.dataset_name = dataset_name

    def read_schema(self, schema_path: str) -> str:
        with open(schema_path, "r") as f:
            schema = f.read()
        return schema

    def prompt_format(self, schema: str, question: str) -> str:
        """Decomposition prompt.

        Resolution chain:
            1. ``prompts.decomposition.<dataset>``
            2. ``prompts.decomposition.general``
            3. Built-in English fallback
        """
        if self.config and hasattr(self.config, "get_prompt_formatted"):
            for key in (self.dataset_name, "general"):
                if not key:
                    continue
                try:
                    prompt = self.config.get_prompt_formatted(
                        "decomposition", key, ontology=schema, question=question
                    )
                except Exception:
                    continue
                if prompt:
                    return prompt

        return self._builtin_decomposition_prompt(schema, question)

    @staticmethod
    def _builtin_decomposition_prompt(schema: str, question: str) -> str:
        return f"""
        You are a professional question decomposition expert specializing
        in multi-hop reasoning. Given the following schema and the question,
        decompose the complex question into 2-3 focused sub-questions.

        CRITICAL REQUIREMENTS:
        1. Each sub-question must be:
        - Specific and focused on a single fact or relationship
        - Answerable independently with the given schema
        - Designed to retrieve relevant knowledge for the final answer
        2. For simple questions (1-2 hop), return the original question as
        a single sub-question
        3. Return a JSON object with "sub_questions" and "involved_types".

        Graph Schema:
        {schema}

        Question: {question}

        Output format:
        {{
        "sub_questions": [{{"sub-question": "..."}}],
        "involved_types": {{"nodes": [], "relations": [], "attributes": []}}
        }}
        """

    def decompose(self, question: str, schema_path: str) -> dict:
        schema = self.read_schema(schema_path)
        prompt = self.prompt_format(schema, question)
        response = self.llm_client.call_api(prompt)
        content = json_repair.loads(response)

        # Ensure backward compatibility - if old format, convert to new format
        if isinstance(content, list):
            content = {
                "sub_questions": content,
                "involved_types": {"nodes": [], "relations": [], "attributes": []},
            }

        return content
