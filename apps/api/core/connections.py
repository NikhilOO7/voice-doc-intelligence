"""
Enhanced database and service connections preserving all original functionality
"""

import logging
import asyncio
from typing import Optional

# Original imports preserved
import redis.asyncio as redis
from qdrant_client import QdrantClient

# Enhanced imports
import pyTigerGraph as tg
from motor.motor_asyncio import AsyncIOMotorClient

from apps.api.core.config import settings

logger = logging.getLogger(__name__)

# Global connection instances (preserving original structure)
_redis_client: Optional[redis.Redis] = None
_qdrant_client: Optional[QdrantClient] = None

# Enhanced connections (new)
_tigergraph_conn: Optional[tg.TigerGraphConnection] = None
_mongodb_client: Optional[AsyncIOMotorClient] = None

async def init_redis():
    """Initialize Redis connection - preserving original functionality"""
    global _redis_client
    try:
        _redis_client = redis.from_url(settings.redis_url)
        await _redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise

async def init_qdrant():
    """Initialize Qdrant connection - preserving original functionality"""
    global _qdrant_client
    try:
        _qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30
        )
        # Test connection
        await asyncio.to_thread(_qdrant_client.get_collections)
        logger.info("✅ Qdrant connected")
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        raise

async def init_tigergraph():
    """Initialize TigerGraph connection - enhanced functionality"""
    global _tigergraph_conn
    try:
        _tigergraph_conn = tg.TigerGraphConnection(
            host=f"http://{settings.tigergraph_host}",
            restppPort=settings.tigergraph_rest_port,
            gsPort=settings.tigergraph_gs_port,
            username=settings.tigergraph_username,
            password=settings.tigergraph_password
        )
        
        # Test connection
        version = _tigergraph_conn.getVersion()
        logger.info(f"✅ TigerGraph connected (version: {version})")
    except Exception as e:
        logger.warning(f"⚠️ TigerGraph connection failed (optional): {e}")
        _tigergraph_conn = None

async def init_mongodb():
    """Initialize MongoDB connection - enhanced functionality"""
    global _mongodb_client
    try:
        mongodb_url = getattr(settings, 'mongodb_url', 'mongodb://localhost:27017')
        _mongodb_client = AsyncIOMotorClient(mongodb_url)
        
        # Test connection
        await _mongodb_client.admin.command('ping')
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB connection failed (optional): {e}")
        _mongodb_client = None

async def init_storage():
    """Initialize storage - preserving original functionality"""
    logger.info("✅ Storage initialized")

# Getter functions (preserving original + adding new)
async def get_redis_client() -> redis.Redis:
    """Get Redis client - preserving original functionality"""
    if not _redis_client:
        await init_redis()
    return _redis_client

async def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client - preserving original functionality"""
    if not _qdrant_client:
        await init_qdrant()
    return _qdrant_client

async def get_tigergraph_conn() -> Optional[tg.TigerGraphConnection]:
    """Get TigerGraph connection - enhanced functionality"""
    if not _tigergraph_conn:
        await init_tigergraph()
    return _tigergraph_conn

async def get_mongodb_client() -> Optional[AsyncIOMotorClient]:
    """Get MongoDB client - enhanced functionality"""
    if not _mongodb_client:
        await init_mongodb()
    return _mongodb_client