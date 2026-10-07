"""Cache cleanup helpers shared by graph construction / reconstruction."""
import glob
import os
import shutil

from utils.logger import logger


async def clear_cache_files(dataset_name: str) -> None:
    """Clear all cache files for a dataset before graph construction."""
    try:
        faiss_cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")

        cache_patterns = [
            f"output/logs/{dataset_name}_*.log",
            f"output/chunks/{dataset_name}_*",
            f"output/graphs/{dataset_name}_*",
        ]
        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    logger.info(f"Cleared cache path: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear {file_path}: {e}")

        logger.info(f"Cache cleanup completed for dataset: {dataset_name}")
    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name}: {e}")