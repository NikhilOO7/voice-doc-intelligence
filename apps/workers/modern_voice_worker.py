#!/usr/bin/env python3
"""
Modern Voice Worker - Standalone Script
Run this to start the voice agent worker
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from livekit.agents import WorkerOptions, cli
from apps.api.services.voice.modern_voice_service import entrypoint
from apps.api.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the voice worker"""
    logger.info("🎙️  Starting Modern Voice Worker...")
    logger.info(f"LiveKit URL: {settings.livekit_url}")

    # Run the worker with CLI
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )


if __name__ == "__main__":
    main()
