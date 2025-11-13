# How to Start Voice Feature

## Important: Two-Process Setup

Due to LiveKit's CLI-based architecture, the voice worker **must run as a separate process**. You'll need **two terminal windows**:
- Terminal 1: Backend API server
- Terminal 2: Voice worker

## Setup Instructions

### Step 1: Install Voice Dependencies

```bash
cd /Users/farheenzubair/Documents/voice-doc-intelligence

# Activate your virtual environment
source venv/bin/activate

# Install LiveKit agents and plugins
pip install livekit-agents==1.1.7
pip install livekit-plugins-deepgram
pip install livekit-plugins-cartesia
pip install livekit-plugins-openai
pip install livekit-plugins-silero
```

### Step 2: Start Your Backend Server

**Terminal 1:**
```bash
cd apps/api
python main.py
```

You'll see:
```
🚀 All services initialized successfully!
🎙️  Voice worker available but must run separately
⚠️  LiveKit agents require CLI-based startup and cannot be embedded in FastAPI
   To enable voice features, run in a separate terminal:
   python apps/workers/enhanced_voice_agent_worker.py dev
```

### Step 3: Start Voice Worker

**Terminal 2** (keep Terminal 1 running):
```bash
cd apps/workers
python enhanced_voice_agent_worker.py dev
```

You'll see:
```
==================================================
Enhanced Voice Agent Worker Starting
Environment: development
Voice Stack: Deepgram + Cartesia + LiveKit
==================================================
Starting LiveKit agent worker...
```

### Step 4: Test Voice

1. Go to your frontend: http://localhost:3000
2. Navigate to the "Voice" tab
3. Click "Connect to Voice Assistant"
4. Allow microphone permissions
5. Start speaking - you should now hear responses!

**Important**: Both Terminal 1 (backend) and Terminal 2 (voice worker) must be running for voice to work!

## Quick Check

**Verify the worker is running:**
```bash
# Check health endpoint to see voice worker status
curl http://localhost:8000/health

# Look for "voice_worker": "running" in the response
```

**Expected response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "qdrant": "healthy",
    "embeddings": "healthy",
    "voice": "healthy",
    "voice_worker": "running"  ← Should say "running"
  }
}
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'livekit.agents'"
**Solution:** Run Step 1 to install dependencies

### Error: "DEEPGRAM_API_KEY is required"
**Solution:** Make sure your `.env` file has valid API keys:
```bash
# Check your .env file
grep DEEPGRAM_API_KEY .env
grep CARTESIA_API_KEY .env
```

### Error: "Cannot connect to Deepgram API"
**Solution:** Check your internet connection and API keys

### Worker starts but voice still doesn't work
**Solution:**
1. Check server logs for errors in the console
2. Restart the backend API: `python apps/api/main.py`
3. Refresh your frontend browser
4. Check browser console for errors (F12)

### How to stop the server gracefully
**Just press Ctrl+C once** - the voice worker will automatically shut down along with the server!

You'll see:
```
Shutting down services...
Stopping voice worker...
Voice worker stopped
✅ Shutdown complete
```

## What's Happening Behind the Scenes

```
User speaks → LiveKit captures audio → Worker receives audio →
Deepgram transcribes → RAG searches documents → OpenAI generates answer →
Cartesia synthesizes speech → LiveKit sends audio back → User hears response
```

The **worker** is the critical component that orchestrates this pipeline.
