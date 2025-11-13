# Voice Document Intelligence - Quick Start Guide

Complete end-to-end setup guide for running the entire Voice Document Intelligence System.

## 🎯 What You're Building

A production-ready voice-enabled document intelligence system with:
- **Smart Document Processing**: Contextual embeddings with 3-level context awareness
- **Voice Interface**: Real-time voice conversations with your documents
- **Chat Interface**: Text-based Q&A with AI
- **Analytics**: Performance metrics and usage tracking

## 📋 Prerequisites

### Required
- **Python 3.11+**: For backend services
- **Node.js 18+**: For frontend
- **Docker & Docker Compose**: For infrastructure services
- **Git**: For version control

### API Keys Required
- **OpenAI API Key**: For LLM and embeddings
- **Deepgram API Key**: For speech-to-text
- **Cartesia API Key**: For text-to-speech

### Optional
- **Voyage AI API Key**: For enhanced embeddings

## 🚀 Step-by-Step Setup

### Step 1: Clone and Navigate

```bash
cd /Users/nikhil007/Documents/voice-doc-intelligence
```

### Step 2: Install Node.js (if not installed)

**macOS:**
```bash
brew install node
```

**Linux:**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**Verify:**
```bash
node --version  # Should be v18+
npm --version
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
# Application
APP_ENV=development
APP_DEBUG=true
LOG_LEVEL=INFO
API_PORT=8000

# Database
POSTGRES_USER=voicedoc
POSTGRES_PASSWORD=voicedoc123
POSTGRES_DB=voice_doc_intel
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=voicedoc123

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# OpenAI (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here

# Deepgram (REQUIRED for voice)
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Cartesia (REQUIRED for voice)
CARTESIA_API_KEY=your_cartesia_api_key_here

# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# Storage
STORAGE_TYPE=local
LOCAL_STORAGE_PATH=./data/uploads

# Optional: Voyage AI for better embeddings
# VOYAGE_API_KEY=your_voyage_api_key_here
EOF
```

**⚠️ IMPORTANT:** Replace the placeholder API keys with your actual keys!

### Step 4: Start Infrastructure Services

```bash
# Start Docker services (Postgres, Redis, Qdrant, LiveKit, etc.)
cd infrastructure/local
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected services:
- ✅ PostgreSQL (port 5432)
- ✅ Redis (port 6379)
- ✅ Qdrant (port 6333)
- ✅ LiveKit (port 7880)
- ✅ MinIO (optional, port 9000)

### Step 5: Set Up Python Backend

```bash
# Return to project root
cd /Users/nikhil007/Documents/voice-doc-intelligence

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install Python dependencies
pip install -r requirements.txt
# OR if you have a pyproject.toml
pip install -e .

# Download required NLP models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 6: Start Backend API

```bash
# Make sure you're in the project root with venv activated
cd apps/api

# Start the FastAPI server
python main.py

# OR use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend should now be running on [http://localhost:8000](http://localhost:8000)

**Verify backend:**
```bash
curl http://localhost:8000/health
```

### Step 7: Start Voice Agent Worker (Optional)

In a **new terminal**:

```bash
cd /Users/nikhil007/Documents/voice-doc-intelligence
source venv/bin/activate

cd apps/workers
python enhanced_voice_agent_worker.py
```

### Step 8: Set Up Frontend

In a **new terminal**:

```bash
cd /Users/nikhil007/Documents/voice-doc-intelligence/apps/web

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend should now be running on [http://localhost:3000](http://localhost:3000)

## ✅ Verify Everything Works

### 1. Check Services

Open these URLs in your browser:

- **Frontend**: http://localhost:3000 (should show the UI)
- **Backend Health**: http://localhost:8000/health (should return healthy status)
- **API Docs**: http://localhost:8000/docs (FastAPI Swagger UI)
- **Qdrant**: http://localhost:6333/dashboard (vector database UI)
- **Redis Insight**: http://localhost:8001 (Redis UI)

### 2. Test Document Upload

1. Go to http://localhost:3000
2. Navigate to "Documents" tab
3. Upload a PDF or DOCX file
4. Wait for processing (status should change from "pending" → "processing" → "completed")

### 3. Test Chat

1. Go to "Chat" tab
2. Type: "What is this document about?"
3. Should receive an AI-generated response with sources

### 4. Test Voice (Optional)

1. Go to "Voice" tab
2. Click "Connect to Voice Assistant"
3. Allow microphone permissions
4. Speak a question
5. Should hear a voice response

### 5. Check Analytics

1. Go to "Analytics" tab
2. Should see document counts, processing metrics, and latency stats

## 🎉 You're All Set!

The complete system is now running:

- ✅ **Frontend**: http://localhost:3000
- ✅ **Backend API**: http://localhost:8000
- ✅ **Voice Worker**: Running in background
- ✅ **Infrastructure**: All Docker services running

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Frontend (Next.js)                  │
│         http://localhost:3000               │
└────────────┬────────────────────────────────┘
             │
             ├─── HTTP/REST ────────┐
             │                      │
             ├─── WebSocket ────────┤
             │                      ▼
┌────────────▼────────────────────────────────┐
│         Backend API (FastAPI)               │
│         http://localhost:8000               │
└────────┬────────────────────────────────────┘
         │
         ├─── PostgreSQL ─── Document Metadata
         ├─── Redis ────────── Caching
         ├─── Qdrant ───────── Vector Storage
         ├─── OpenAI ───────── LLM & Embeddings
         ├─── Deepgram ─────── Speech-to-Text
         ├─── Cartesia ─────── Text-to-Speech
         └─── LiveKit ──────── Voice Infrastructure
```

## 🔧 Common Issues & Solutions

### Backend Won't Start

**Error**: "Database connection failed"
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

**Error**: "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Won't Start

**Error**: "Port 3000 already in use"
```bash
# Kill existing process
lsof -ti:3000 | xargs kill -9
```

**Error**: "Module not found"
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Voice Not Working

**Error**: "Failed to connect to voice service"
```bash
# Check LiveKit is running
docker-compose ps livekit

# Restart LiveKit
docker-compose restart livekit
```

### Document Processing Fails

**Error**: "Processing status: failed"
- Check backend logs for specific error
- Verify file format is supported (PDF, DOCX, TXT)
- Check file size is under 100MB

## 📝 Usage Tips

### Upload Documents
- Supported formats: PDF, DOCX, DOC, TXT
- Max file size: 100MB
- Enable "enhanced processing" for better results

### Query Documents
- Use natural language questions
- Select appropriate context level:
  - **Local**: Fast, focused search
  - **Document**: Search entire document
  - **Global**: Search across all documents

### Voice Features
- Speak clearly and naturally
- Wait for the "listening" indicator
- Voice works best in quiet environments

## 🛑 Shutting Down

### Stop All Services

```bash
# Stop frontend (Ctrl+C in frontend terminal)
# Stop backend (Ctrl+C in backend terminal)
# Stop voice worker (Ctrl+C in worker terminal)

# Stop Docker services
cd infrastructure/local
docker-compose down

# Or keep data and just stop:
docker-compose stop
```

### Restart Everything

```bash
# Start Docker services
cd infrastructure/local
docker-compose up -d

# Start backend (in one terminal)
cd apps/api
source ../../venv/bin/activate
python main.py

# Start frontend (in another terminal)
cd apps/web
npm run dev
```

## 📚 Next Steps

1. **Upload Sample Documents**: Try different document types
2. **Explore Chat**: Ask various questions to test RAG quality
3. **Try Voice**: Test the voice assistant
4. **Check Analytics**: Monitor system performance
5. **Read Documentation**: Check individual README files for details

## 🆘 Getting Help

- Check logs in terminals where services are running
- Review `apps/api/main.py` logs for backend errors
- Check browser console (F12) for frontend errors
- Verify all API keys are correct in `.env`

## 🎓 Learn More

- **Backend**: See `apps/api/README.md` (if exists)
- **Frontend**: See `apps/web/README.md`
- **Infrastructure**: See `infrastructure/local/README.md` (if exists)

---

**Congratulations!** You now have a fully functional voice-enabled document intelligence system running locally! 🚀
