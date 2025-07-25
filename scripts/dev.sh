# Updated scripts/dev.sh
#!/bin/bash
# scripts/dev.sh - Updated with Voice Agent

set -e

echo "🚀 Starting Voice Document Intelligence System (Development Mode)"

# Start infrastructure if not running
echo "📦 Starting infrastructure services..."
cd infrastructure/local
docker-compose up -d
cd ../..

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 15

# Check service health
echo "🔍 Checking service health..."
services=("tigergraph:9000" "qdrant:6333" "redis-stack:6379" "postgres:5432" "livekit:7880")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if nc -z localhost $port; then
        echo "✅ $name is running on port $port"
    else
        echo "❌ $name is not accessible on port $port"
    fi
done

# Terminal 1: Backend API
echo "🐍 Starting FastAPI backend..."
tmux new-session -d -s backend "cd apps/api && poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000"

# Terminal 2: Frontend
echo "🌐 Starting Astro frontend..."
tmux new-session -d -s frontend "cd apps/web && pnpm dev"

# Terminal 3: Voice Agent Worker
echo "🎙️ Starting Voice Agent Worker..."
tmux new-session -d -s voice-agent "cd . && poetry run python apps/workers/voice_agent_worker.py"

echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📍 Service URLs:"
echo "  - Frontend:           http://localhost:4321"
echo "  - Backend API:        http://localhost:8000/api/v1/docs"
echo "  - LiveKit:            http://localhost:7880"
echo "  - TigerGraph Studio:  http://localhost:14240"
echo "  - MinIO Console:      http://localhost:9002"
echo "  - RedisInsight:       http://localhost:8001"
echo ""
echo "🎙️ Voice Features:"
echo "  - Voice interface available in frontend"
echo "  - LiveKit Agent running for document intelligence"
echo "  - Real-time STT/TTS with contextual document search"
echo ""
echo "📋 View logs:"
echo "  - Backend:      tmux attach-session -t backend"
echo "  - Frontend:     tmux attach-session -t frontend" 
echo "  - Voice Agent:  tmux attach-session -t voice-agent"
echo ""
echo "🛑 Stop all: docker-compose -f infrastructure/local/docker-compose.yml down && tmux kill-server"