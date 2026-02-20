"""
gRPC API implementation for text embeddings.

This server provides the same embedding functionality as the REST API
but uses Protocol Buffers and gRPC (HTTP/2 native).
"""

import sys
import os
import time
import logging
from concurrent import futures

import grpc
from prometheus_client import Counter, Histogram, start_http_server

# Add parent directory to path to import shared module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.model import EmbeddingModel

# Import generated protobuf code
import embedding_pb2
import embedding_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'grpc_api_requests_total',
    'Total number of requests',
    ['method', 'status']
)

REQUEST_LATENCY = Histogram(
    'grpc_api_request_duration_seconds',
    'Request latency in seconds',
    ['method']
)

INFERENCE_LATENCY = Histogram(
    'grpc_api_inference_duration_seconds',
    'Model inference latency in seconds'
)


class EmbeddingServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """gRPC servicer implementation for embedding service."""

    def __init__(self):
        """Initialize the embedding model."""
        self.model = EmbeddingModel()
        logger.info("gRPC EmbeddingServicer initialized")

    def Embed(self, request, context):
        """Generate embedding for a single text."""
        start_time = time.time()

        try:
            if not request.text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text cannot be empty")
                REQUEST_COUNT.labels(method='Embed', status='INVALID_ARGUMENT').inc()
                return embedding_pb2.EmbedResponse()

            # Perform inference
            inference_start = time.time()
            embedding = self.model.embed(request.text)
            inference_time = (time.time() - inference_start) * 1000  # Convert to ms

            request_duration = time.time() - start_time

            # Record metrics
            REQUEST_COUNT.labels(method='Embed', status='OK').inc()
            REQUEST_LATENCY.labels(method='Embed').observe(request_duration)
            INFERENCE_LATENCY.observe(inference_time / 1000)  # Convert to seconds

            # Build response
            response = embedding_pb2.EmbedResponse(
                embedding=embedding,
                dimension=self.model.get_embedding_dimension(),
                inference_time_ms=inference_time
            )

            return response

        except Exception as e:
            logger.error(f"Error in Embed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            REQUEST_COUNT.labels(method='Embed', status='INTERNAL').inc()
            return embedding_pb2.EmbedResponse()

    def EmbedBatch(self, request, context):
        """Generate embeddings for multiple texts (batch)."""
        start_time = time.time()

        try:
            if not request.texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Texts list cannot be empty")
                REQUEST_COUNT.labels(method='EmbedBatch', status='INVALID_ARGUMENT').inc()
                return embedding_pb2.BatchEmbedResponse()

            # Perform inference
            inference_start = time.time()
            embeddings = self.model.embed_batch(list(request.texts))
            inference_time = (time.time() - inference_start) * 1000  # Convert to ms

            request_duration = time.time() - start_time

            # Record metrics
            REQUEST_COUNT.labels(method='EmbedBatch', status='OK').inc()
            REQUEST_LATENCY.labels(method='EmbedBatch').observe(request_duration)
            INFERENCE_LATENCY.observe(inference_time / 1000)  # Convert to seconds

            # Build response
            embedding_vectors = [
                embedding_pb2.EmbeddingVector(values=emb)
                for emb in embeddings
            ]

            response = embedding_pb2.BatchEmbedResponse(
                embeddings=embedding_vectors,
                count=len(embeddings),
                dimension=self.model.get_embedding_dimension(),
                inference_time_ms=inference_time
            )

            return response

        except Exception as e:
            logger.error(f"Error in EmbedBatch: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            REQUEST_COUNT.labels(method='EmbedBatch', status='INTERNAL').inc()
            return embedding_pb2.BatchEmbedResponse()

    def HealthCheck(self, request, context):
        """Health check endpoint."""
        try:
            REQUEST_COUNT.labels(method='HealthCheck', status='OK').inc()

            response = embedding_pb2.HealthCheckResponse(
                status="healthy",
                model=self.model.model_name,
                embedding_dimension=self.model.get_embedding_dimension()
            )

            return response

        except Exception as e:
            logger.error(f"Error in HealthCheck: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            REQUEST_COUNT.labels(method='HealthCheck', status='INTERNAL').inc()
            return embedding_pb2.HealthCheckResponse()


def serve():
    """Start the gRPC server."""
    # Start Prometheus metrics server on port 8001
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001")

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServicer(), server
    )

    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()

    logger.info("gRPC server started on port 50051 (HTTP/2)")
    logger.info("Prometheus metrics available at http://localhost:8001/metrics")

    server.wait_for_termination()


if __name__ == '__main__':
    serve()
