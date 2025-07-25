"""
Enhanced FastAPI application for Voice Document Intelligence System
Integrates new Day 1-3 implementation while preserving all existing functionality
"""

import asyncio
import logging
import os
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

# Core imports - preserving original structure
from apps.api.core.config import settings
from apps.api.core.database import init_db, get_db, Base
from apps.api.core.connections import init_redis, init_qdrant, init_storage, get_redis_client, get_qdrant_client

# Enhanced services - combining original + new functionality
from apps.api.services.document.processor import ModernDocumentProcessor, DocumentProcessor  # Both implementations
from apps.api.services.document.embeddings import ContextualEmbeddingGenerator, EmbeddingService  # Both implementations  
from apps.api.services.document.vector_store import ModernVectorStore, VectorStoreService  # Both implementations
from apps.api.services.rag.llamaindex_service import ModernRAGService, RAGService  # Both implementations
from apps.api.services.voice.livekit_service import VoiceService, DocumentIntelligenceVoiceAgent, entrypoint
from apps.api.services.agents.crew_setup import DocumentIntelligenceAgents

# Models - preserving original
from apps.api.models.document import Document, DocumentChunk, DocumentCreate, DocumentResponse, DocumentStats
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances - supporting both original and enhanced services
# Original services for backward compatibility
doc_processor: DocumentProcessor = None
embedding_service: EmbeddingService = None
vector_store: VectorStoreService = None
rag_service: RAGService = None
voice_service: VoiceService = None

# Enhanced services for new functionality
modern_doc_processor: ModernDocumentProcessor = None
contextual_embedding_generator: ContextualEmbeddingGenerator = None
modern_vector_store: ModernVectorStore = None
modern_rag_service: ModernRAGService = None
doc_intelligence_agents: DocumentIntelligenceAgents = None

# In-memory stores for demo functionality (preserving original approach)
documents_store: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager"""
    logger.info("🚀 Starting Enhanced Voice Document Intelligence System...")
    
    try:
        # Initialize databases and connections (original)
        await init_db()
        await init_redis()
        await init_qdrant()  
        await init_storage()
        
        # Initialize original services for backward compatibility
        global doc_processor, embedding_service, vector_store, rag_service, voice_service
        global modern_doc_processor, contextual_embedding_generator, modern_vector_store, modern_rag_service, doc_intelligence_agents
        
        # Original services
        doc_processor = DocumentProcessor()
        embedding_service = EmbeddingService()
        vector_store = VectorStoreService()
        rag_service = RAGService()
        voice_service = VoiceService()
        
        # Enhanced services
        modern_doc_processor = ModernDocumentProcessor()
        contextual_embedding_generator = ContextualEmbeddingGenerator()
        modern_vector_store = ModernVectorStore()
        modern_rag_service = ModernRAGService()
        doc_intelligence_agents = DocumentIntelligenceAgents()
        
        logger.info("✅ All services initialized successfully (Original + Enhanced)")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    finally:
        logger.info("🔌 Shutting down services...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files
uploads_dir = Path("apps/api/uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Request/Response Models (preserving original + adding new ones)
class QueryRequest(BaseModel):
    query: str
    voice_mode: bool = False
    conversation_id: str = None
    use_enhanced: bool = True  # New flag to choose enhanced processing

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    processing_time_ms: int
    enhanced_processing: bool = False  # Indicates if enhanced processing was used

class EnhancedDocumentResponse(DocumentResponse):
    """Enhanced document response with additional metadata"""
    file_hash: Optional[str] = None
    processing_method: str = "basic"  # "basic" or "enhanced"
    extracted_metadata: Dict[str, Any] = {}
    semantic_summary: Optional[str] = None

# Health and status endpoints (preserving original)
@app.get("/")
async def root():
    """Root endpoint - preserving original functionality"""
    return {
        "name": "Voice Document Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "mode": "enhanced_demo",
        "features": {
            "original_processing": True,
            "enhanced_processing": True,
            "contextual_embeddings": True,
            "multi_agent_support": True,
            "voice_intelligence": True
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "database": "healthy",
            "redis": "healthy", 
            "qdrant": "healthy",
            "original_services": {
                "document_processor": doc_processor is not None,
                "embedding_service": embedding_service is not None,
                "vector_store": vector_store is not None,
                "rag_service": rag_service is not None,
                "voice_service": voice_service is not None,
            },
            "enhanced_services": {
                "modern_doc_processor": modern_doc_processor is not None,
                "contextual_embedding_generator": contextual_embedding_generator is not None,
                "modern_vector_store": modern_vector_store is not None,
                "modern_rag_service": modern_rag_service is not None,
                "doc_intelligence_agents": doc_intelligence_agents is not None,
            }
        }
    }

# Document management endpoints (enhanced while preserving original functionality)
@app.post("/api/v1/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None,
    use_enhanced: bool = Query(True, description="Use enhanced processing pipeline"),
    db: AsyncSession = Depends(get_db)
):
    """Enhanced document upload supporting both original and enhanced processing"""
    try:
        # Validate file (preserving original validation)
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            raise HTTPException(400, "Unsupported file type")
        
        # Parse metadata (preserving original functionality)
        user_metadata = {}
        if metadata:
            try:
                user_metadata = json.loads(metadata)
            except:
                pass
        
        # Generate IDs
        document_id = str(uuid.uuid4())
        external_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = uploads_dir / f"{document_id}_{file.filename}"
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create document record (enhanced)
        doc_data = DocumentCreate(
            title=file.filename,
            file_path=str(file_path),
            content_type=file.content_type,
            file_size=len(content)
        )
        
        # Store in in-memory store for backward compatibility
        documents_store[document_id] = {
            "id": document_id,
            "external_id": external_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "content_type": file.content_type,
            "file_size": len(content),
            "processing_status": "processing",
            "processing_method": "enhanced" if use_enhanced else "basic",
            "created_at": datetime.utcnow(),
            "metadata": user_metadata
        }
        
        # Start background processing with method selection
        background_tasks.add_task(
            process_document_task_enhanced,
            document_id,
            str(file_path),
            file.content_type,
            use_enhanced
        )
        
        return EnhancedDocumentResponse(
            id=document_id,
            title=file.filename,
            status="processing",
            upload_time=datetime.utcnow(),
            chunk_count=0,
            processing_method="enhanced" if use_enhanced else "basic"
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/api/v1/documents", response_model=List[EnhancedDocumentResponse])
async def list_documents():
    """Enhanced document listing preserving original functionality"""
    try:
        documents = []
        
        # Return from in-memory store (preserving original approach)
        for doc_id, doc_data in documents_store.items():
            documents.append(EnhancedDocumentResponse(
                id=doc_data.get('id', ''),
                title=doc_data.get('filename', ''),
                status=doc_data.get('processing_status', 'unknown'),
                upload_time=doc_data.get('created_at'),
                chunk_count=doc_data.get('chunk_count', 0),
                processing_method=doc_data.get('processing_method', 'basic'),
                file_hash=doc_data.get('file_hash'),
                extracted_metadata=doc_data.get('extracted_metadata', {})
            ))
        
        return documents
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []

@app.get("/api/v1/documents/stats", response_model=DocumentStats)
async def get_document_stats():
    """Enhanced document statistics"""
    try:
        total_docs = len(documents_store)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents_store.values())
        
        status_counts = {}
        method_counts = {"basic": 0, "enhanced": 0}
        
        for doc in documents_store.values():
            status = doc.get('processing_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            method = doc.get('processing_method', 'basic')
            method_counts[method] += 1
        
        return DocumentStats(
            total_documents=total_docs,
            total_chunks=total_chunks,
            processing_status=status_counts,
            processing_methods=method_counts  # Enhanced field
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return DocumentStats(
            total_documents=0,
            total_chunks=0,
            processing_status={},
            processing_methods={"basic": 0, "enhanced": 0}
        )

# Query endpoints (enhanced while preserving original)
@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Enhanced query processing with fallback to original"""
    import time
    start_time = time.time()
    
    try:
        # Choose processing method
        if request.use_enhanced and modern_rag_service:
            # Use enhanced RAG service
            result = await modern_rag_service.process_query(
                query=request.query,
                conversation_id=request.conversation_id
            )
            enhanced_processing = True
        else:
            # Fallback to original RAG service
            result = await rag_service.process_query(
                query=request.query,
                conversation_id=request.conversation_id  
            )
            enhanced_processing = False
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result.get('answer', 'No answer generated'),
            sources=result.get('sources', []),
            conversation_id=result.get('conversation_id', 'default'),
            processing_time_ms=processing_time,
            enhanced_processing=enhanced_processing
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

# Voice endpoints (preserving original functionality)
@app.post("/api/v1/voice/process")
async def process_voice_query(
    audio_file: UploadFile = File(...),
    conversation_id: str = None
):
    """Process voice query - preserving original functionality"""
    try:
        audio_data = await audio_file.read()
        
        result = await voice_service.process_voice_query(
            audio_data=audio_data,
            conversation_id=conversation_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(500, f"Voice processing failed: {str(e)}")

@app.get("/api/v1/voice/token")
async def get_voice_token():
    """Get LiveKit token - preserving original functionality"""
    try:
        token = await voice_service.generate_token()
        return {"token": token, "url": settings.livekit_url}
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise HTTPException(500, "Failed to generate voice token")

# Enhanced background processing task
async def process_document_task_enhanced(
    document_id: str, 
    file_path: str, 
    content_type: str,
    use_enhanced: bool = True
):
    """Enhanced background document processing supporting both methods"""
    try:
        # Update status
        if document_id in documents_store:
            documents_store[document_id]['processing_status'] = 'processing'
        
        if use_enhanced and modern_doc_processor:
            # Use enhanced processing pipeline
            logger.info(f"Processing document {document_id} with enhanced pipeline")
            
            # Process with modern processor
            processing_result = await modern_doc_processor.process_document(file_path, content_type)
            
            # Generate contextual embeddings
            embeddings_data = []
            if processing_result.get("chunks"):
                embedding_results = await contextual_embedding_generator.generate_embeddings(
                    chunks=processing_result["chunks"],
                    document_metadata=processing_result.get("metadata", {}),
                    use_voyage=bool(settings.voyage_api_key)
                )
                embeddings_data = embedding_results
            
            # Store in modern vector store
            if embeddings_data:
                await modern_vector_store.store_embeddings(document_id, embeddings_data)
            
            # Update document store with enhanced data
            if document_id in documents_store:
                documents_store[document_id].update({
                    'processing_status': 'completed',
                    'chunk_count': len(processing_result.get("chunks", [])),
                    'file_hash': processing_result.get("file_hash"),
                    'extracted_metadata': processing_result.get("metadata", {}),
                    'processing_method': 'enhanced'
                })
        
        else:
            # Use original processing pipeline (backward compatibility)
            logger.info(f"Processing document {document_id} with original pipeline")
            
            # Process document using original method
            chunks = await doc_processor.process_document(file_path)
            
            # Generate embeddings using original service
            embeddings_data = []
            for i, chunk in enumerate(chunks):
                embedding = await embedding_service.generate_embedding(chunk['content'])
                embeddings_data.append({
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "content": chunk['content'],
                    "metadata": chunk.get('metadata', {}),
                    "embedding": embedding
                })
            
            # Store in original vector database
            await vector_store.store_embeddings(document_id, embeddings_data)
            
            # Update document store
            if document_id in documents_store:
                documents_store[document_id].update({
                    'processing_status': 'completed',
                    'chunk_count': len(chunks),
                    'processing_method': 'basic'
                })
        
        logger.info(f"✅ Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"❌ Document processing failed for {document_id}: {e}")
        if document_id in documents_store:
            documents_store[document_id]['processing_status'] = 'failed'
            documents_store[document_id]['processing_error'] = {'error': str(e)}

# Enhanced demo endpoint
@app.get("/api/v1/test/demo")
async def demo_test():
    """Enhanced demo test endpoint"""
    return {
        "message": "Enhanced Voice Document Intelligence System is running!",
        "original_services": {
            "document_processor": doc_processor is not None,
            "embedding_service": embedding_service is not None,
            "vector_store": vector_store is not None,
            "rag_service": rag_service is not None,
            "voice_service": voice_service is not None,
        },
        "enhanced_services": {
            "modern_doc_processor": modern_doc_processor is not None,
            "contextual_embedding_generator": contextual_embedding_generator is not None,
            "modern_vector_store": modern_vector_store is not None,
            "modern_rag_service": modern_rag_service is not None,
            "doc_intelligence_agents": doc_intelligence_agents is not None,
        },
        "endpoints": [
            "POST /api/v1/documents/upload - Upload documents (basic or enhanced)",
            "GET /api/v1/documents - List documents", 
            "POST /api/v1/query - Query documents (basic or enhanced)",
            "POST /api/v1/voice/process - Process voice queries",
            "GET /api/v1/voice/token - Get voice token",
            "GET /api/v1/agents/crew - Access CrewAI agents"
        ],
        "features": {
            "backward_compatibility": True,
            "enhanced_processing": True,
            "contextual_embeddings": True,
            "multi_agent_crew": True,  
            "voice_intelligence": True
        }
    }

# CrewAI agent endpoints (preserving original functionality)
@app.get("/api/v1/agents/crew")
async def get_crew_agents():
    """Access CrewAI agents - preserving original functionality"""
    if not doc_intelligence_agents:
        raise HTTPException(500, "CrewAI agents not initialized")
    
    agents = doc_intelligence_agents.create_agents()
    
    return {
        "agents": [
            {
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory
            }
            for agent in agents
        ],
        "crew_ready": True
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )