# apps/api/models/document.py
from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey, Text, Float, Boolean, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..core.database import Base

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "doc_intel"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(255), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_type = Column(String(100))
    file_size = Column(Integer)
    file_hash = Column(String(64))
    
    # Processing status
    processing_status = Column(String(50), default="pending")
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    processing_error = Column(JSON)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    extracted_metadata = Column(JSON, default=dict)
    user_metadata = Column(JSON, default=dict)
    
    # Versioning
    version = Column(Integer, default=1)
    parent_document_id = Column(UUID(as_uuid=True), ForeignKey("doc_intel.documents.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime(timezone=True))
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = {"schema": "doc_intel"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("doc_intel.documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64))
    token_count = Column(Integer)
    
    # Position
    start_char = Column(Integer)
    end_char = Column(Integer)
    start_page = Column(Integer)
    end_page = Column(Integer)
    
    # Structure
    section_path = Column(ARRAY(Text))
    heading_level = Column(Integer)
    is_table = Column(Boolean, default=False)
    is_list = Column(Boolean, default=False)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    embeddings = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    contextual_embeddings = relationship("ContextualEmbedding", back_populates="chunk", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="chunk")

class ContextualEmbedding(Base):
    __tablename__ = "contextual_embeddings"
    __table_args__ = {"schema": "doc_intel"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("doc_intel.document_chunks.id"), nullable=False)
    
    context_type = Column(String(50), nullable=False)
    context_window = Column(Integer, default=3)
    
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    embedding_dimension = Column(Integer, nullable=False)
    
    # Note: Actual embedding stored in Qdrant, this stores metadata
    embedding_id = Column(String(255))  # ID in vector database
    
    context_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    chunk = relationship("DocumentChunk", back_populates="contextual_embeddings")

class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = {"schema": "doc_intel"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("doc_intel.documents.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("doc_intel.document_chunks.id"))
    
    entity_type = Column(String(100), nullable=False)
    entity_value = Column(Text, nullable=False)
    entity_normalized = Column(Text)
    
    confidence_score = Column(Float)
    extraction_method = Column(String(50))
    
    start_offset = Column(Integer)
    end_offset = Column(Integer)
    
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="entities")
    chunk = relationship("DocumentChunk", back_populates="entities")