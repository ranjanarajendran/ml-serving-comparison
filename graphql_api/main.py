"""
GraphQL API implementation using Strawberry with HTTP/2 support via Hypercorn.

This API provides text embedding functionality using GraphQL queries and mutations.
"""

from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

from schema import schema

# Initialize FastAPI app
app = FastAPI(
    title="ML Embedding GraphQL API",
    description="GraphQL API for text embeddings with HTTP/2 support",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'graphql_api_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'graphql_api_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint']
)

# Create GraphQL router
graphql_app = GraphQLRouter(schema)

# Mount GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Embedding GraphQL API",
        "version": "1.0.0",
        "protocol": "GraphQL over HTTP/2",
        "endpoints": {
            "graphql": "/graphql",
            "graphiql": "/graphql (browser)",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(endpoint='/health', status='200').inc()

    return {
        "status": "healthy",
        "api": "graphql"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


# To run with HTTP/2 support, use Hypercorn:
# hypercorn main:app --bind 0.0.0.0:8001
