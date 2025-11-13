# Voice Worker Integration Summary

## What Changed

The voice agent worker is now **integrated into your main FastAPI server** and runs automatically as a background task.

## Changes Made to `apps/api/main.py`

### 1. Added Global Variables (Lines 133-135)
```python
# Voice worker task
voice_worker_task: Optional[asyncio.Task] = None
voice_worker_shutdown_event: Optional[asyncio.Event] = None
```

### 2. Added Voice Worker Function (Lines 137-165)
```python
async def run_voice_worker():
    """Run the LiveKit voice agent worker as a background task"""
    # Automatically checks if dependencies are installed
    # Validates API keys
    # Runs the worker using livekit.agents.cli
```

### 3. Updated Lifespan Manager (Lines 219-242)
**On Startup:**
- Creates background task: `voice_worker_task = asyncio.create_task(run_voice_worker())`
- Worker starts automatically if dependencies are available

**On Shutdown:**
- Cancels the worker task gracefully
- Waits for clean shutdown

### 4. Enhanced Health Check (Lines 335-369)
Added `voice_worker` status to the `/health` endpoint:
- `"running"` - Worker is active
- `"disabled"` - Dependencies not installed
- `"stopped"` - Worker stopped
- `"failed"` - Worker encountered an error

## How It Works

```
┌─────────────────────────────────────────────┐
│   Start: python apps/api/main.py           │
└────────────────┬────────────────────────────┘
                 │
                 ├─── Initialize Database
                 ├─── Initialize Redis/Qdrant
                 ├─── Initialize Services
                 │
                 ├─── ✨ Start Voice Worker (Background)
                 │    │
                 │    ├─ Check if livekit-agents installed
                 │    ├─ Validate API keys
                 │    └─ Run LiveKit agent worker
                 │
                 └─── FastAPI Server Running
                      │
                      ├─ API Endpoints (Port 8000)
                      └─ Voice Worker (Background)
                           │
                           └─ Processes voice in real-time

Ctrl+C
  │
  ├─── Shutdown Signal
  ├─── Cancel Voice Worker Task
  ├─── Wait for Clean Shutdown
  └─── Exit ✅
```

## Benefits

✅ **Single Process** - No need to run separate terminal for voice worker
✅ **Automatic Startup** - Worker starts when server starts
✅ **Graceful Shutdown** - Worker stops when server stops (Ctrl+C)
✅ **Fault Tolerant** - Server works even if voice dependencies missing
✅ **Health Monitoring** - Check worker status via `/health` endpoint
✅ **Clean Logs** - All logs in one place

## User Experience

**Before:**
```bash
# Terminal 1
python apps/api/main.py

# Terminal 2
python apps/workers/enhanced_voice_agent_worker.py

# To stop: Ctrl+C in both terminals
```

**After:**
```bash
# Just one terminal
python apps/api/main.py

# To stop: Ctrl+C (stops everything)
```

## Installation Steps for Voice

```bash
# Install voice dependencies (one-time)
pip install livekit-agents==1.1.7
pip install livekit-plugins-deepgram
pip install livekit-plugins-cartesia
pip install livekit-plugins-openai
pip install livekit-plugins-silero

# Start server (voice worker starts automatically)
python apps/api/main.py
```

## Verification

```bash
# Check if voice worker is running
curl http://localhost:8000/health | jq '.services.voice_worker'

# Should return: "running"
```

## Fallback Behavior

If voice dependencies are NOT installed:
```
⚠️  livekit-agents not installed. Voice worker disabled.
   Install with: pip install livekit-agents livekit-plugins-deepgram livekit-plugins-cartesia
```

Server continues running normally, just without voice features.

## Error Handling

The worker runs in a background task with proper error handling:
- Import errors → Warning logged, server continues
- Missing API keys → Warning logged, server continues
- Worker crash → Error logged, task marked as failed
- All errors are non-blocking to the main server

## Code Quality

- ✅ Type hints used (`Optional[asyncio.Task]`)
- ✅ Proper async/await patterns
- ✅ Graceful shutdown with task cancellation
- ✅ Comprehensive logging
- ✅ Health check integration
- ✅ Non-blocking background execution
