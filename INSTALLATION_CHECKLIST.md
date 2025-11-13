# Installation Checklist

Use this checklist to ensure everything is set up correctly for running the Voice Document Intelligence System end-to-end.

## ✅ Pre-Installation

### System Requirements
- [ ] macOS, Linux, or Windows with WSL2
- [ ] 8GB+ RAM
- [ ] 10GB+ free disk space
- [ ] Stable internet connection

### Software Installed
- [ ] **Docker Desktop** installed and running
  ```bash
  docker --version  # Should show 20.10+
  docker-compose --version  # Should show 1.29+
  ```

- [ ] **Python 3.11+** installed
  ```bash
  python3 --version  # Should show 3.11+
  pip --version
  ```

- [ ] **Node.js 18+** installed
  ```bash
  node --version  # Should show v18+
  npm --version  # Should show 9+
  ```

- [ ] **Git** installed
  ```bash
  git --version
  ```

### API Keys Obtained
- [ ] **OpenAI API Key** - https://platform.openai.com/api-keys
- [ ] **Deepgram API Key** - https://console.deepgram.com/
- [ ] **Cartesia API Key** - https://cartesia.ai/
- [ ] (Optional) **Voyage AI API Key** - https://www.voyageai.com/

## ✅ Project Setup

### 1. Environment Configuration
- [ ] Created `.env` file in project root
- [ ] Added all required API keys to `.env`
- [ ] Configured database credentials in `.env`
- [ ] Set correct ports in `.env` (if different from defaults)

### 2. Infrastructure Services
```bash
cd infrastructure/local
docker-compose up -d
```

- [ ] **PostgreSQL** running (port 5432)
  ```bash
  docker-compose ps postgres  # Should show "Up"
  ```

- [ ] **Redis** running (port 6379)
  ```bash
  docker-compose ps redis-stack  # Should show "Up"
  ```

- [ ] **Qdrant** running (port 6333)
  ```bash
  docker-compose ps qdrant  # Should show "Up"
  curl http://localhost:6333/health  # Should return OK
  ```

- [ ] **LiveKit** running (port 7880)
  ```bash
  docker-compose ps livekit  # Should show "Up"
  ```

### 3. Python Backend Setup
```bash
cd /Users/farheenzubair/Documents/voice-doc-intelligence
```

- [ ] Virtual environment created
  ```bash
  python3 -m venv venv
  ```

- [ ] Virtual environment activated
  ```bash
  source venv/bin/activate  # macOS/Linux
  # OR
  venv\Scripts\activate  # Windows
  ```

- [ ] Python dependencies installed
  ```bash
  pip install -e .
  # OR
  pip install -r requirements.txt
  ```

- [ ] NLP models downloaded
  ```bash
  python -c "import spacy; spacy.cli.download('en_core_web_sm')"
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
  ```

### 4. Frontend Setup
```bash
cd apps/web
```

- [ ] Node dependencies installed
  ```bash
  npm install
  ```

- [ ] `.env.local` file exists
  ```bash
  ls -la .env.local
  ```

- [ ] Build succeeds
  ```bash
  npm run build  # Should complete without errors
  ```

## ✅ Service Verification

### Backend API
```bash
cd apps/api
source ../../venv/bin/activate
python main.py
```

- [ ] Backend starts without errors
- [ ] Backend accessible at http://localhost:8000
- [ ] Health check passes:
  ```bash
  curl http://localhost:8000/health
  # Should return: {"status": "healthy", ...}
  ```
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] Can see FastAPI Swagger UI

### Frontend
```bash
cd apps/web
npm run dev
```

- [ ] Frontend starts without errors
- [ ] Frontend accessible at http://localhost:3000
- [ ] Page loads without console errors (check browser F12)
- [ ] Health indicator shows green/healthy

### Voice Worker (Optional)
```bash
cd apps/workers
source ../../venv/bin/activate
python enhanced_voice_agent_worker.py
```

- [ ] Worker starts without errors
- [ ] Worker connects to LiveKit
- [ ] Health endpoint responds: http://localhost:8081/health

## ✅ Functional Testing

### 1. Document Upload
- [ ] Navigate to http://localhost:3000
- [ ] Go to "Documents" tab
- [ ] Drag and drop a PDF file
- [ ] File uploads successfully
- [ ] Status changes to "processing"
- [ ] Status eventually changes to "completed"
- [ ] Document appears in list with metadata

### 2. Chat Interface
- [ ] Go to "Chat" tab
- [ ] Type a question: "What is this document about?"
- [ ] Get a response within 2-5 seconds
- [ ] Response includes source citations
- [ ] Sources show relevance scores
- [ ] Latency metrics displayed

### 3. Voice Interface (Optional)
- [ ] Go to "Voice" tab
- [ ] Click "Connect to Voice Assistant"
- [ ] Browser asks for microphone permission
- [ ] Grant permission
- [ ] Connection succeeds
- [ ] Speak: "Tell me about my documents"
- [ ] Voice response plays back
- [ ] Can see "listening" → "thinking" → "speaking" states

### 4. Analytics Dashboard
- [ ] Go to "Analytics" tab
- [ ] See document count (should be > 0)
- [ ] See chunk statistics
- [ ] See latency metrics
- [ ] Charts and metrics display correctly

### 5. Real-time Updates
- [ ] Upload a new document
- [ ] Watch status change in real-time
- [ ] Check notifications appear
- [ ] Browser notification pops up (if enabled)

## ✅ Infrastructure Checks

### Database Connections
- [ ] **Postgres accessible**:
  ```bash
  docker exec -it voice-doc-postgres psql -U voicedoc -d voice_doc_intel -c "\dt"
  ```

- [ ] **Redis accessible**:
  ```bash
  docker exec -it voice-doc-redis redis-cli -a voicedoc123 ping
  # Should return: PONG
  ```

- [ ] **Qdrant accessible**:
  ```bash
  curl http://localhost:6333/collections
  ```

### Monitoring Dashboards
- [ ] **Qdrant Dashboard**: http://localhost:6333/dashboard
- [ ] **Redis Insight**: http://localhost:8001
- [ ] **Prometheus** (if enabled): http://localhost:9090
- [ ] **Grafana** (if enabled): http://localhost:3000

## ✅ Performance Checks

### Latency Targets
Check in Analytics dashboard:
- [ ] STT latency < 200ms
- [ ] LLM latency < 1000ms
- [ ] TTS latency < 150ms
- [ ] RAG latency < 500ms
- [ ] Total pipeline < 2000ms

### System Resources
```bash
docker stats --no-stream
```

- [ ] PostgreSQL using < 500MB RAM
- [ ] Redis using < 200MB RAM
- [ ] Qdrant using < 1GB RAM
- [ ] Overall CPU < 80%

## ✅ Error Handling

### Test Error Scenarios
- [ ] Upload invalid file format → Shows error message
- [ ] Upload file > 100MB → Shows size error
- [ ] Query with empty message → Button disabled
- [ ] Disconnect backend → Frontend shows error
- [ ] Invalid API key → Helpful error message

## ✅ Browser Compatibility

Test on:
- [ ] Chrome/Edge (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)

## 🔧 Troubleshooting

If any step fails, refer to:
1. [QUICKSTART.md](QUICKSTART.md) - Detailed setup
2. [README.md](README.md) - Project overview
3. [apps/web/SETUP.md](apps/web/SETUP.md) - Frontend-specific
4. Backend logs in terminal
5. Frontend logs in browser console (F12)
6. Docker logs: `docker-compose logs <service>`

## 📝 Notes

### Important Ports
- **3000**: Frontend (Next.js)
- **8000**: Backend API (FastAPI)
- **8001**: Redis Insight
- **8080**: Temporal UI (optional)
- **8081**: Voice worker health
- **5432**: PostgreSQL
- **6333**: Qdrant
- **6379**: Redis
- **7880**: LiveKit
- **9000**: MinIO (optional)
- **9090**: Prometheus (optional)
- **9091**: Voice worker metrics

### Data Storage
When using local storage (`STORAGE_TYPE=local`):
- Documents stored in: `./data/uploads/`
- Embeddings in: Qdrant container volume
- Metadata in: PostgreSQL container volume

## ✅ Final Verification

All systems green when:
- [ ] All Docker containers running
- [ ] Backend API responding
- [ ] Frontend loading
- [ ] Can upload documents
- [ ] Can query documents
- [ ] Can use voice (if configured)
- [ ] Analytics showing data
- [ ] No errors in logs

## 🎉 Success!

If all items are checked, your Voice Document Intelligence System is fully operational!

### Next Steps:
1. Upload your documents
2. Try different query types
3. Explore voice features
4. Monitor analytics
5. Customize as needed

### Get Help:
- Check logs for errors
- Review documentation
- Verify API keys
- Ensure all services running
- Check firewall settings

---

**Ready to start using the system! 🚀**
