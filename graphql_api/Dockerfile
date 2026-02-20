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

# Copy GraphQL API code
COPY schema.py .
COPY main.py .

# Expose port
EXPOSE 8001

# Run with Hypercorn for HTTP/2 support
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8001"]
