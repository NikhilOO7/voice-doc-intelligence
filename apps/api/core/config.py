# apps/api/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import Optional, Dict, Any
import os
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = "Voice Document Intelligence"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=False, env="APP_DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Settings
    api_prefix: str = "/api/v1"
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: list[str] = Field(default=["http://localhost:4321"], env="CORS_ORIGINS")
    
    # Database URLs
    postgres_url: str = Field(
        default="postgresql+asyncpg://docintell:docintell123@localhost:5432/document_intelligence",
        env="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Vector Databases
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, env="QDRANT_GRPC_PORT")
    
    # TigerGraph
    tigergraph_host: str = Field(default="localhost", env="TIGERGRAPH_HOST")
    tigergraph_rest_port: int = Field(default=9000, env="TIGERGRAPH_REST_PORT")
    tigergraph_gs_port: int = Field(default=14240, env="TIGERGRAPH_GS_PORT")
    tigergraph_username: str = Field(default="tigergraph", env="TIGERGRAPH_USERNAME")
    tigergraph_password: str = Field(default="tigergraph123", env="TIGERGRAPH_PASSWORD")
    
    # AI/LLM Settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    
    # Voyage AI (Better embeddings)
    voyage_api_key: Optional[str] = Field(default=None, env="VOYAGE_API_KEY")
    voyage_model: str = Field(default="voyage-3-large", env="VOYAGE_MODEL")
    
    # Voice Processing
    livekit_url: str = Field(default="ws://localhost:7880", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="devkey", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="secret", env="LIVEKIT_API_SECRET")
    
    # Cartesia (Modern voice)
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    
    # Storage
    minio_endpoint: str = Field(default="localhost:9001", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="documents", env="MINIO_BUCKET")

    # Document Processing Settings (new - enhanced functionality)
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_file_size: int = Field(default=100*1024*1024, env="MAX_FILE_SIZE")  # 100MB

    # Processing Pipeline Settings (new)
    use_enhanced_processing: bool = Field(default=True, env="USE_ENHANCED_PROCESSING")
    enable_contextual_embeddings: bool = Field(default=True, env="ENABLE_CONTEXTUAL_EMBEDDINGS")
    enable_multi_agent_crew: bool = Field(default=True, env="ENABLE_MULTI_AGENT_CREW")

    # Performance Settings (new)
    embedding_batch_size: int = Field(default=10, env="EMBEDDING_BATCH_SIZE")
    max_concurrent_processing: int = Field(default=5, env="MAX_CONCURRENT_PROCESSING")
    
    # Temporal
    temporal_host: str = Field(default="localhost:7233", env="TEMPORAL_HOST")
    temporal_namespace: str = Field(default="default", env="TEMPORAL_NAMESPACE")
    
    # Pulsar
    pulsar_url: str = Field(default="pulsar://localhost:6650", env="PULSAR_URL")
    pulsar_topic_prefix: str = Field(default="doc-intel", env="PULSAR_TOPIC_PREFIX")
    
    # Security
    jwt_secret: str = Field(default="your-secret-key", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def tigergraph_url(self) -> str:
        return f"http://{self.tigergraph_host}:{self.tigergraph_rest_port}"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()