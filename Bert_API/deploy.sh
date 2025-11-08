#!/bin/bash

# Stop and remove old containers named bert-api
echo "🛑 Stopping any old bert-api containers..."
docker stop bert-api-container 2>/dev/null || true
docker rm bert-api-container 2>/dev/null || true

# Build new image (with host network for reliable DNS)
echo "🐳 Building Docker image..."
docker build --network=host -t bert-api .

# Run new container
echo "🚀 Starting new container..."
docker run -d --name bert-api-container -p 8080:8080 bert-api

# Show latest logs once (avoid hanging)
echo "📜 Showing last 50 log lines..."
docker logs --tail 50 bert-api-container
