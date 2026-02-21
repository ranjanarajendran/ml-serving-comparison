"""
Locust load testing script for ML embedding APIs.

This script tests REST, gRPC, and GraphQL APIs under various load scenarios.
"""

import json
import random
from locust import HttpUser, task, between, events
import grpc
from google.protobuf import json_format

# Import test data
from test_data import get_random_text, get_batch_texts

# Import gRPC generated code (will be generated from proto)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'grpc_api')))


class RESTAPIUser(HttpUser):
    """Load test user for REST API."""

    host = "http://localhost:8000"
    wait_time = between(1, 3)

    @task(3)
    def embed_single(self):
        """Test single text embedding endpoint."""
        text = get_random_text('medium')
        payload = {"text": text}

        with self.client.post(
            "/embed",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def embed_batch(self):
        """Test batch embedding endpoint."""
        texts = get_batch_texts(count=5, length='medium')
        payload = {"texts": texts}

        with self.client.post(
            "/embed/batch",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health")


class GraphQLAPIUser(HttpUser):
    """Load test user for GraphQL API."""

    host = "http://localhost:8002"
    wait_time = between(1, 3)

    @task(3)
    def embed_single(self):
        """Test single text embedding via GraphQL."""
        text = get_random_text('medium')

        query = """
        mutation Embed($input: EmbedInput!) {
            embed(input: $input) {
                embedding
                dimension
                inferenceTimeMs
            }
        }
        """

        variables = {
            "input": {
                "text": text
            }
        }

        payload = {
            "query": query,
            "variables": variables
        }

        with self.client.post(
            "/graphql",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "errors" in data:
                    response.failure(f"GraphQL errors: {data['errors']}")
                else:
                    response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def embed_batch(self):
        """Test batch embedding via GraphQL."""
        texts = get_batch_texts(count=5, length='medium')

        query = """
        mutation EmbedBatch($input: BatchEmbedInput!) {
            embedBatch(input: $input) {
                embeddings
                count
                dimension
                inferenceTimeMs
            }
        }
        """

        variables = {
            "input": {
                "texts": texts
            }
        }

        payload = {
            "query": query,
            "variables": variables
        }

        with self.client.post(
            "/graphql",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "errors" in data:
                    response.failure(f"GraphQL errors: {data['errors']}")
                else:
                    response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health check query."""
        query = """
        query {
            health {
                status
                model
                embeddingDimension
            }
        }
        """

        payload = {"query": query}

        self.client.post("/graphql", json=payload)


# gRPC test user would require more complex setup with grpc channels
# For now, we'll focus on REST and GraphQL load tests
# gRPC testing can be done separately or with a custom gRPC locust implementation

class MixedAPIUser(HttpUser):
    """Load test user that randomly picks between REST and GraphQL APIs."""

    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Randomly choose API for this user
        self.api_type = random.choice(['rest', 'graphql'])

        if self.api_type == 'rest':
            self.host = "http://localhost:8000"
        else:
            self.host = "http://localhost:8002"

    @task
    def test_embed(self):
        """Test embedding endpoint based on selected API."""
        text = get_random_text('medium')

        if self.api_type == 'rest':
            payload = {"text": text}
            self.client.post("/embed", json=payload)
        else:  # graphql
            query = """
            mutation Embed($input: EmbedInput!) {
                embed(input: $input) {
                    embedding
                    dimension
                    inferenceTimeMs
                }
            }
            """
            variables = {"input": {"text": text}}
            payload = {"query": query, "variables": variables}
            self.client.post("/graphql", json=payload)


# Custom test scenarios can be defined here
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("=" * 60)
    print("Starting ML Serving API Load Tests")
    print("=" * 60)
    print(f"Target hosts:")
    print(f"  REST API:    http://localhost:8000")
    print(f"  GraphQL API: http://localhost:8002")
    print(f"  gRPC API:    localhost:50051")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Cleanup after tests."""
    print("=" * 60)
    print("Load tests completed")
    print("=" * 60)
