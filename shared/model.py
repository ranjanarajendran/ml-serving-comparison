"""
Shared embedding model wrapper used by all API implementations.

This ensures fair comparison by using the exact same model and inference code
across REST, gRPC, and GraphQL APIs.
"""

import time
from typing import List
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for SentenceTransformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully: {model_name}")

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for input text.

        Args:
            text: Input text string

        Returns:
            List of floats representing the embedding vector (384 dimensions)
        """
        start_time = time.time()
        embedding = self.model.encode(text).tolist()
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.debug(f"Inference time: {inference_time:.2f}ms for text length: {len(text)}")

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch inference).

        Args:
            texts: List of input text strings

        Returns:
            List of embedding vectors
        """
        start_time = time.time()
        embeddings = self.model.encode(texts).tolist()
        inference_time = (time.time() - start_time) * 1000

        logger.debug(f"Batch inference time: {inference_time:.2f}ms for {len(texts)} texts")

        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return self.model.get_sentence_embedding_dimension()
