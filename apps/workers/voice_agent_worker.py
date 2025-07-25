# apps/workers/voice_agent_worker.py
"""
Voice Agent Worker for Document Intelligence System
Runs LiveKit agents that handle voice interactions with document content
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from livekit.agents import WorkerOptions, cli
from apps.api.services.voice.livekit_service import entrypoint
from apps.api.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('voice_agent.log')
    ]
)

logger = logging.getLogger("voice-agent-worker")

def main():
    """Main entry point for the voice agent worker"""
    
    # Validate configuration
    required_env_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY", 
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    logger.info("Starting Document Intelligence Voice Agent Worker...")
    logger.info(f"LiveKit URL: {settings.livekit_url}")
    logger.info(f"Environment: {settings.environment}")
    
    # Configure worker options
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        # Enable development mode for hot reloading
        dev_mode=settings.environment == "development",
        # Worker configuration
        max_retry_count=3,
        worker_type="voice-agent",
        # Logging
        log_level=settings.log_level.lower(),
    )
    
    try:
        # Run the worker
        cli.run_app(worker_options)
    except KeyboardInterrupt:
        logger.info("Voice agent worker stopped by user")
    except Exception as e:
        logger.error(f"Voice agent worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()