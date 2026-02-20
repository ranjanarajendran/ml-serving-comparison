"""
REST API implementation using FastAPI with HTTP/2 support via Hypercorn.

This API provides text embedding endpoints using the shared SentenceTransformer model.
"""

import sys
import os
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Add parent directory to path to import shared module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.model import EmbeddingModel

# Initialize FastAPI app
app = FastAPI(
    title="ML Embedding REST API",
    description="REST API for text embeddings with HTTP/2 support",
    version="1.0.0"
)

# Initialize embedding model
model = EmbeddingModel()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'rest_api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'rest_api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

INFERENCE_LATENCY = Histogram(
    'rest_api_inference_duration_seconds',
    'Model inference latency in seconds'
)


# Request/Response models
class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed", min_length=1)


class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="384-dimensional embedding vector")
    dimension: int = Field(..., description="Embedding dimension")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1)


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    count: int = Field(..., description="Number of embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_dimension: int


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Embedding REST API",
        "version": "1.0.0",
        "protocol": "REST over HTTP/2",
        "endpoints": {
            "health": "/health",
            "embed": "/embed",
            "batch_embed": "/embed/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()

    return HealthResponse(
        status="healthy",
        model=model.model_name,
        embedding_dimension=model.get_embedding_dimension()
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embedding for a single text."""
    start_time = time.time()

    try:
        inference_start = time.time()
        embedding = model.embed(request.text)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms

        request_duration = time.time() - start_time

        # Record metrics
        REQUEST_COUNT.labels(method='POST', endpoint='/embed', status='200').inc()
        REQUEST_LATENCY.labels(method='POST', endpoint='/embed').observe(request_duration)
        INFERENCE_LATENCY.observe(inference_time / 1000)  # Convert to seconds

        return EmbedResponse(
            embedding=embedding,
            dimension=model.get_embedding_dimension(),
            inference_time_ms=inference_time
        )
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/embed', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest):
    """Generate embeddings for multiple texts (batch inference)."""
    start_time = time.time()

    try:
        inference_start = time.time()
        embeddings = model.embed_batch(request.texts)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms

        request_duration = time.time() - start_time

        # Record metrics
        REQUEST_COUNT.labels(method='POST', endpoint='/embed/batch', status='200').inc()
        REQUEST_LATENCY.labels(method='POST', endpoint='/embed/batch').observe(request_duration)
        INFERENCE_LATENCY.observe(inference_time / 1000)  # Convert to seconds

        return BatchEmbedResponse(
            embeddings=embeddings,
            count=len(embeddings),
            dimension=model.get_embedding_dimension(),
            inference_time_ms=inference_time
        )
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/embed/batch', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


# To run with HTTP/2 support, use Hypercorn:
# hypercorn main:app --bind 0.0.0.0:8000
