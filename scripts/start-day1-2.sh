#!/bin/bash
# scripts/start-day1-2.sh

set -e

echo "🚀 Starting Voice Document Intelligence System (Day 1-2)"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "📦 Starting infrastructure services..."
cd infrastructure/local
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
services=("tigergraph:9000" "qdrant:6333" "redis-stack:6379" "postgres:5432" "minio:9001")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if nc -z localhost $port; then
        echo "✅ $name is running on port $port"
    else
        echo "❌ $name is not accessible on port $port"
    fi
done

# Initialize TigerGraph schema (if needed)
echo "🔧 Initializing TigerGraph schema..."
# Add TigerGraph schema initialization here

# Create MinIO bucket
echo "🪣 Creating MinIO bucket..."
docker exec -it local-minio-1 mc alias set local http://localhost:9001 minioadmin minioadmin
docker exec -it local-minio-1 mc mb local/documents || true

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd ../../
poetry install

# Run database migrations
echo "🗄️ Setting up database..."
poetry run python -c "
import asyncio
from apps.api.core.database import engine, Base
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(create_tables())
"

# Start the API server
echo "🚀 Starting API server..."
cd apps/api
poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000 &

echo "✅ Day 1-2 Implementation is running!"
echo ""
echo "📍 Service URLs:"
echo "  - API Documentation: http://localhost:8000/api/v1/docs"
echo "  - TigerGraph Studio: http://localhost:14240"
echo "  - MinIO Console: http://localhost:9002 (minioadmin/minioadmin)"
echo "  - RedisInsight: http://localhost:8001"
echo ""
echo "📝 Next steps:"
echo "  1. Upload a document: POST http://localhost:8000/api/v1/documents/upload"
echo "  2. Check processing status: GET http://localhost:8000/api/v1/documents/{document_id}"
echo "  3. List documents: GET http://localhost:8000/api/v1/documents"
echo ""
echo "To stop all services: docker-compose down"