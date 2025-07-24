# apps/api/core/connections.py
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pyTigerGraph as tg
from minio import Minio
from temporalio.client import Client as TemporalClient
import pulsar
from typing import Optional
import logging

from .config import settings

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Singleton manager for all external connections"""
    
    _instance = None
    _redis_client: Optional[redis.Redis] = None
    _qdrant_client: Optional[QdrantClient] = None
    _tigergraph_conn: Optional[tg.TigerGraphConnection] = None
    _minio_client: Optional[Minio] = None
    _temporal_client: Optional[TemporalClient] = None
    _pulsar_client: Optional[pulsar.Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis connection with connection pooling"""
        if self._redis_client is None:
            self._redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
            )
            # Test connection
            await self._redis_client.ping()
            logger.info("Redis connection established")
        return self._redis_client
    
    def get_qdrant(self) -> QdrantClient:
        """Get Qdrant client with gRPC for better performance"""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=True,
            )
            logger.info("Qdrant client initialized")
        return self._qdrant_client
    
    def get_tigergraph(self) -> tg.TigerGraphConnection:
        """Get TigerGraph connection"""
        if self._tigergraph_conn is None:
            self._tigergraph_conn = tg.TigerGraphConnection(
                host=settings.tigergraph_host,
                graphname="DocumentIntelligence",
                username=settings.tigergraph_username,
                password=settings.tigergraph_password,
                restppPort=settings.tigergraph_rest_port,
                gsPort=settings.tigergraph_gs_port,
            )
            # Generate token
            self._tigergraph_conn.getToken()
            logger.info("TigerGraph connection established")
        return self._tigergraph_conn
    
    def get_minio(self) -> Minio:
        """Get MinIO client for object storage"""
        if self._minio_client is None:
            self._minio_client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=False,  # Set to True in production with HTTPS
            )
            # Ensure bucket exists
            if not self._minio_client.bucket_exists(settings.minio_bucket):
                self._minio_client.make_bucket(settings.minio_bucket)
            logger.info("MinIO client initialized")
        return self._minio_client
    
    async def get_temporal(self) -> TemporalClient:
        """Get Temporal client"""
        if self._temporal_client is None:
            self._temporal_client = await TemporalClient.connect(
                settings.temporal_host,
                namespace=settings.temporal_namespace,
            )
            logger.info("Temporal client connected")
        return self._temporal_client
    
    def get_pulsar(self) -> pulsar.Client:
        """Get Pulsar client for event streaming"""
        if self._pulsar_client is None:
            self._pulsar_client = pulsar.Client(
                settings.pulsar_url,
                operation_timeout_seconds=30,
            )
            logger.info("Pulsar client connected")
        return self._pulsar_client
    
    async def close_all(self):
        """Close all connections gracefully"""
        if self._redis_client:
            await self._redis_client.close()
        if self._qdrant_client:
            self._qdrant_client.close()
        if self._pulsar_client:
            self._pulsar_client.close()
        logger.info("All connections closed")

# Global connection manager instance
connections = ConnectionManager()