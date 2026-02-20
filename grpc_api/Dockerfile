FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared model wrapper
COPY ../shared/model.py /app/shared/

# Copy proto file and generate Python code
COPY embedding.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. embedding.proto

# Copy gRPC server code
COPY server.py .

# Expose ports (50051 for gRPC, 8001 for Prometheus metrics)
EXPOSE 50051 8001

# Run gRPC server
CMD ["python", "server.py"]
