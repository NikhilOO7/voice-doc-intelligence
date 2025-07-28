#!/bin/bash
# start-docker.sh - Helper script to start Docker and services

echo "🐳 Checking Docker status..."

# Function to check if Docker is running
check_docker() {
    docker info >/dev/null 2>&1
    return $?
}

# Start Docker if not running
if ! check_docker; then
    echo "❌ Docker is not running. Starting Docker Desktop..."
    
    # Try to start Docker Desktop on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Docker
        echo "⏳ Waiting for Docker to start..."
        
        # Wait up to 60 seconds for Docker to start
        for i in {1..60}; do
            if check_docker; then
                echo "✅ Docker is now running!"
                break
            fi
            echo -n "."
            sleep 1
        done
        
        if ! check_docker; then
            echo ""
            echo "❌ Docker failed to start. Please start Docker Desktop manually."
            exit 1
        fi
    else
        echo "❌ Please start Docker manually and run this script again."
        exit 1
    fi
else
    echo "✅ Docker is already running!"
fi

# Now start the services
echo ""
echo "🚀 Starting Voice Document Intelligence services..."

cd ./infrastructure/local

# Pull latest images
echo "📦 Pulling Docker images..."
docker-compose pull

# Start services
echo "🔧 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."

services=(
    "postgres:5432"
    "redis-stack:6379"
    "qdrant:6333"
    "minio:9000"
    "livekit:7880"
)

all_healthy=true
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    container_name="voice-doc-${name}"
    
    if docker ps | grep -q $container_name; then
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $name is running on port $port"
        else
            echo "⚠️  $name container is running but port $port is not accessible"
            all_healthy=false
        fi
    else
        echo "❌ $name container is not running"
        all_healthy=false
    fi
done

echo ""
if $all_healthy; then
    echo "🎉 All services are running!"
    echo ""
    echo "📍 Service URLs:"
    echo "  - PostgreSQL:    localhost:5432"
    echo "  - Redis:         localhost:6379"
    echo "  - RedisInsight:  http://localhost:8001"
    echo "  - Qdrant:        http://localhost:6333"
    echo "  - MinIO:         http://localhost:9000"
    echo "  - MinIO Console: http://localhost:9002"
    echo "  - LiveKit:       http://localhost:7880"
    echo ""
    echo "Run './scripts/dev.sh' to start the application!"
else
    echo "⚠️  Some services failed to start. Check the logs with:"
    echo "  docker-compose logs -f [service-name]"
fi