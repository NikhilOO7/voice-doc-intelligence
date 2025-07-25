#!/bin/bash
# scripts/setup.sh

set -e

echo "🚀 Setting up Voice Document Intelligence System..."

# Check prerequisites
command -v python3.11 >/dev/null 2>&1 || { echo "Python 3.11 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js 20+ required"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "pnpm required: npm install -g pnpm"; exit 1; }

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Install Node dependencies
echo "📦 Installing Node dependencies..."
pnpm install

# Setup environment
cp .env.example .env
echo "⚠️  Please configure your .env file with API keys"

# Start infrastructure
echo "🐳 Starting Docker services..."
cd infrastructure/local
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Initialize databases
echo "🗄️ Initializing databases..."
poetry run python scripts/init_databases.py

echo "✅ Setup complete!"
echo ""
echo "To start development:"
echo "  ./scripts/dev.sh"