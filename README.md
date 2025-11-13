# Voice Document Intelligence System

A production-ready AI-powered voice-enabled document intelligence platform with advanced contextual embeddings and real-time voice interaction.

![System Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Node](https://img.shields.io/badge/node-18+-green)
![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

## 🌟 Features

### 📄 Intelligent Document Processing
- **Multi-format Support**: PDF, DOCX, DOC, TXT
- **Smart Chunking**: Preserves semantic boundaries and document structure
- **3-Level Contextual Embeddings**:
  - **Local Context**: Surrounding chunks (50% weight)
  - **Document Context**: Document structure and themes (30% weight)
  - **Global Context**: Cross-document relationships (20% weight)
- **Rich Metadata Extraction**: Named entities, keywords, section hierarchy

### 🎤 Voice Interface
- **Ultra-Low Latency**: Target <200ms end-to-end
- **Advanced Voice Stack**:
  - Speech-to-Text: Deepgram Nova-3
  - Text-to-Speech: Cartesia Sonic
  - Infrastructure: LiveKit WebRTC
- **Natural Conversations**: Context-aware dialogue with documents

### 💬 Chat Interface
- **Contextual Q&A**: Ask questions in natural language
- **Source Attribution**: See which chunks informed each answer
- **Multi-level Search**: Local, document, or global context
- **Conversation Memory**: Maintains dialogue context

### 🤖 Multi-Agent Architecture
Five specialized AI agents working together:
1. **Document Agent**: Intelligent processing and chunking
2. **Voice Agent**: Real-time speech interaction
3. **Query Agent**: Intent recognition and enhancement
4. **Context Agent**: Multi-level contextual search
5. **Analytics Agent**: Usage patterns and insights

### 📊 Analytics Dashboard
- Document processing statistics
- Performance metrics (latency, success rates)
- System health monitoring
- Usage analytics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js + React)                │
│              TypeScript • Tailwind • LiveKit                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
            │ HTTP/REST │ WebSocket │ LiveKit/WebRTC
            │           │           │
┌───────────▼───────────▼───────────▼─────────────────────────┐
│                Backend API (FastAPI)                         │
│         Python • Async • Multi-Agent System                  │
└───────────┬───────────────────────────────────────┬─────────┘
            │                                       │
    ┌───────┴───────┐                       ┌──────┴──────┐
    │ Infrastructure │                       │  AI Services │
    └───────┬───────┘                       └──────┬──────┘
            │                                       │
    ┌───────▼────────────────┐              ┌──────▼──────────────┐
    │ • PostgreSQL           │              │ • OpenAI GPT-4      │
    │ • Redis                │              │ • Deepgram Nova-3   │
    │ • Qdrant (Vectors)     │              │ • Cartesia Sonic    │
    │ • LiveKit Server       │              │ • Sentence Trans.   │
    └────────────────────────┘              └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Docker & Docker Compose**
- **API Keys**:
  - OpenAI API key (required)
  - Deepgram API key (for voice)
  - Cartesia API key (for voice)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd voice-doc-intelligence

# Run the setup script
./start-dev.sh

# Follow the on-screen instructions
```

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md)

### Manual Setup

1. **Start Infrastructure**
   ```bash
   cd infrastructure/local
   docker-compose up -d
   ```

2. **Start Backend**
   ```bash
   cd apps/api
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ../..
   python main.py
   ```

3. **Start Frontend**
   ```bash
   cd apps/web
   npm install
   npm run dev
   ```

4. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
voice-doc-intelligence/
├── apps/
│   ├── api/                      # FastAPI backend
│   │   ├── main.py              # Main application
│   │   ├── core/                # Configuration & database
│   │   ├── models/              # Data models
│   │   ├── routers/             # API routes
│   │   └── services/            # Business logic
│   │       ├── agents/          # Multi-agent system
│   │       ├── document/        # Document processing
│   │       ├── voice/           # Voice services
│   │       └── rag/            # RAG implementation
│   ├── web/                     # Next.js frontend
│   │   ├── app/                # Next.js app directory
│   │   ├── components/         # React components
│   │   ├── lib/                # Utilities & API client
│   │   └── types/              # TypeScript types
│   └── workers/                # Background workers
│       └── enhanced_voice_agent_worker.py
├── infrastructure/
│   └── local/
│       ├── docker-compose.yml  # Local development stack
│       └── livekit.yaml        # LiveKit configuration
├── data/                       # Data storage (local mode)
├── .env                        # Environment variables
├── pyproject.toml             # Python dependencies
├── start-dev.sh               # Development startup script
├── QUICKSTART.md              # Detailed setup guide
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# AI Services
OPENAI_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here

# Database
POSTGRES_USER=voicedoc
POSTGRES_PASSWORD=voicedoc123
POSTGRES_DB=voice_doc_intel

# Redis
REDIS_PASSWORD=voicedoc123

# Application
APP_ENV=development
API_PORT=8000
STORAGE_TYPE=local
```

See [QUICKSTART.md](QUICKSTART.md) for complete configuration details.

## 📚 Usage

### 1. Upload Documents

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf" \
  -F "use_enhanced=true"

# Via Web UI
# Go to http://localhost:3000 → Documents tab → Drag & drop
```

### 2. Query Documents

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main topics?",
    "context_level": "document",
    "use_enhanced": true
  }'

# Via Web UI
# Go to Chat tab → Type your question
```

### 3. Use Voice Interface

1. Go to Voice tab in the web UI
2. Click "Connect to Voice Assistant"
3. Allow microphone permissions
4. Speak your question naturally

## 🎯 Key Technologies

### Backend
- **FastAPI**: Modern async Python web framework
- **SQLAlchemy**: Database ORM
- **Qdrant**: Vector database for embeddings
- **Redis**: Caching and real-time features
- **OpenAI**: GPT-4 for LLM capabilities
- **CrewAI**: Multi-agent orchestration
- **spaCy**: NLP and entity extraction
- **sentence-transformers**: Local embeddings

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **LiveKit React**: Voice/video components
- **Axios**: HTTP client
- **Recharts**: Analytics visualization

### Infrastructure
- **Docker**: Containerization
- **PostgreSQL**: Relational database
- **Redis Stack**: In-memory database
- **Qdrant**: Vector similarity search
- **LiveKit**: Real-time communication
- **MinIO**: S3-compatible object storage

## 📈 Performance

- **Document Processing**: <5s for typical PDF
- **Query Latency**: <2s end-to-end
- **Voice Pipeline**:
  - STT: <150ms (Deepgram)
  - LLM: <800ms (GPT-4)
  - TTS: <100ms (Cartesia)
  - Target Total: <200ms

## 🧪 Testing

```bash
# Backend tests
cd apps/api
pytest

# Frontend tests
cd apps/web
npm test

# End-to-end tests
npm run test:e2e
```

## 📖 Documentation

- [Quick Start Guide](QUICKSTART.md) - Complete setup instructions
- [Frontend Setup](apps/web/SETUP.md) - Frontend-specific guide
- [Frontend README](apps/web/README.md) - Frontend documentation
- [Agent Documentation](apps/api/services/agents/README.md) - Multi-agent system

## 🔍 API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000 (frontend)
lsof -ti:3000 | xargs kill -9
```

**Docker Services Won't Start**
```bash
# Reset Docker services
cd infrastructure/local
docker-compose down -v
docker-compose up -d
```

**Voice Not Working**
- Verify LiveKit is running: `docker-compose ps livekit`
- Check microphone permissions in browser
- Ensure API keys are set correctly

See [QUICKSTART.md](QUICKSTART.md) for more troubleshooting tips.

## 📊 Monitoring

Access monitoring dashboards:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Redis Insight**: http://localhost:8001
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 🔒 Security

- API keys stored in environment variables
- CORS configured for local development
- File upload validation and size limits
- SQL injection prevention via ORM
- XSS protection in frontend

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** - GPT-4 and embeddings
- **Deepgram** - Speech-to-text
- **Cartesia** - Text-to-speech
- **LiveKit** - Real-time communication infrastructure
- **CrewAI** - Multi-agent framework
- **Qdrant** - Vector database

## 📧 Support

For issues and questions:
1. Check [QUICKSTART.md](QUICKSTART.md) for setup help
2. Review existing GitHub issues
3. Create a new issue with detailed information

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics with ML insights
- [ ] Mobile application
- [ ] Collaboration features
- [ ] Knowledge graph visualization
- [ ] Custom agent creation framework
- [ ] Enterprise SSO integration
- [ ] Advanced security features

---

**Built with ❤️ for intelligent document interaction**

*Voice Document Intelligence v2.0*
