"""
GraphQL schema definition for text embedding service.

Defines GraphQL types, queries, and mutations for the embedding API.
"""

import time
from typing import List
import strawberry

import sys
import os

# Add parent directory to path to import shared module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.model import EmbeddingModel

# Initialize embedding model
model = EmbeddingModel()


@strawberry.type
class EmbedResult:
    """Result type for single text embedding."""
    embedding: List[float]
    dimension: int
    inference_time_ms: float


@strawberry.type
class BatchEmbedResult:
    """Result type for batch text embeddings."""
    embeddings: List[List[float]]
    count: int
    dimension: int
    inference_time_ms: float


@strawberry.type
class HealthCheck:
    """Health check result."""
    status: str
    model: str
    embedding_dimension: int


@strawberry.input
class EmbedInput:
    """Input type for single text embedding."""
    text: str


@strawberry.input
class BatchEmbedInput:
    """Input type for batch text embeddings."""
    texts: List[str]


@strawberry.type
class Query:
    """GraphQL queries."""

    @strawberry.field
    def health(self) -> HealthCheck:
        """Health check query."""
        return HealthCheck(
            status="healthy",
            model=model.model_name,
            embedding_dimension=model.get_embedding_dimension()
        )


@strawberry.type
class Mutation:
    """GraphQL mutations."""

    @strawberry.mutation
    def embed(self, input: EmbedInput) -> EmbedResult:
        """Generate embedding for a single text."""
        if not input.text:
            raise ValueError("Text cannot be empty")

        start_time = time.time()
        embedding = model.embed(input.text)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        return EmbedResult(
            embedding=embedding,
            dimension=model.get_embedding_dimension(),
            inference_time_ms=inference_time
        )

    @strawberry.mutation
    def embed_batch(self, input: BatchEmbedInput) -> BatchEmbedResult:
        """Generate embeddings for multiple texts (batch)."""
        if not input.texts:
            raise ValueError("Texts list cannot be empty")

        start_time = time.time()
        embeddings = model.embed_batch(input.texts)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        return BatchEmbedResult(
            embeddings=embeddings,
            count=len(embeddings),
            dimension=model.get_embedding_dimension(),
            inference_time_ms=inference_time
        )


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
