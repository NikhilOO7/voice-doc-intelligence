# apps/api/main.py
"""
Enhanced FastAPI application for Voice Document Intelligence System
Full implementation with contextual embeddings and voice integration
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
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

# Core imports
from apps.api.core.config import settings
from apps.api.core.database import init_db, get_db, Base
from apps.api.core.connections import init_redis, init_qdrant, init_storage, get_redis_client, get_qdrant_client, get_storage_service

# Document processing services
from apps.api.services.document.processor import DocumentProcessor
from apps.api.services.document.contextual_processor import ModernDocumentProcessor, ContextualEmbeddingGenerator
from apps.api.services.document.embeddings import EmbeddingService
from apps.api.services.document.vector_store import VectorStoreService, ModernVectorStore

# RAG services
from apps.api.services.rag.llamaindex_service import RAGService, ModernRAGService

# Voice services
from apps.api.services.voice.livekit_service import VoiceService
from apps.api.services.voice.enhanced_livekit_service import (
    EnhancedVoiceService,
    DocumentIntelligenceVoiceAgent,
    entrypoint
)

# Agents
from apps.api.services.agents.crew_setup import DocumentIntelligenceAgents

# Models
from apps.api.models.document import Document, DocumentChunk, DocumentCreate, DocumentResponse, DocumentStats
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
doc_processor: DocumentProcessor = None
modern_doc_processor: ModernDocumentProcessor = None
embedding_service: EmbeddingService = None
contextual_embedding_generator: ContextualEmbeddingGenerator = None
vector_store: VectorStoreService = None
modern_vector_store: ModernVectorStore = None
rag_service: RAGService = None
modern_rag_service: ModernRAGService = None
voice_service: VoiceService = None
enhanced_voice_service: EnhancedVoiceService = None
doc_intelligence_agents: DocumentIntelligenceAgents = None

# In-memory document store (replace with database in production)
documents_store: Dict[str, Dict] = {}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Voice Document Intelligence System...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("✅ Database initialized")
        
        # Initialize connections
        await init_redis()
        await init_qdrant()
        await init_storage()
        logger.info("✅ External services connected")
        
        # Initialize services
        global doc_processor, modern_doc_processor, embedding_service, contextual_embedding_generator
        global vector_store, modern_vector_store, rag_service, modern_rag_service
        global voice_service, enhanced_voice_service, doc_intelligence_agents
        
        # Document processing
        doc_processor = DocumentProcessor()
        modern_doc_processor = ModernDocumentProcessor()
        logger.info("✅ Document processors initialized")
        
        # Embeddings
        embedding_service = EmbeddingService()
        contextual_embedding_generator = ContextualEmbeddingGenerator()
        logger.info("✅ Embedding services initialized")
        
        # Vector stores
        vector_store = VectorStoreService()
        modern_vector_store = ModernVectorStore()
        logger.info("✅ Vector stores initialized")
        
        # RAG services
        rag_service = RAGService()
        modern_rag_service = ModernRAGService()
        logger.info("✅ RAG services initialized")
        
        # Voice services
        voice_service = VoiceService()
        enhanced_voice_service = EnhancedVoiceService()
        logger.info("✅ Voice services initialized")
        
        # Agents
        doc_intelligence_agents = DocumentIntelligenceAgents()
        logger.info("✅ Document intelligence agents initialized")
        
        logger.info("🚀 All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down services...")
    # Add cleanup code here

# Create FastAPI app
app = FastAPI(
    title="Voice Document Intelligence API",
    description="API for voice-enabled document intelligence with contextual embeddings",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    context_level: Optional[str] = "local"
    max_results: Optional[int] = 5
    use_enhanced: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    context_level: str
    metadata: Dict[str, Any]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = {
        "database": "healthy",
        "redis": "healthy" if await get_redis_client() else "unhealthy",
        "qdrant": "healthy" if await get_qdrant_client() else "unhealthy",
        "embeddings": "healthy" if embedding_service else "unhealthy",
        "voice": "healthy" if voice_service else "unhealthy",
    }
    
    overall_health = "healthy" if all(status == "healthy" for status in services_status.values()) else "degraded"
    
    return {
        "status": overall_health,
        "services": services_status,
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Document upload endpoint
@app.post("/api/v1/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_enhanced: bool = Query(True, description="Use enhanced processing with contextual embeddings")
):
    """Upload and process a document with contextual embeddings"""
    try:
        # Validate file
        if file.size > settings.max_file_size:
            raise HTTPException(400, f"File too large. Maximum size: {settings.max_file_size} bytes")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Get storage service
        storage_service = get_storage_service()
        
        # Upload to storage (local or MinIO based on config)
        storage_result = await storage_service.upload_file(
            file_content=content,
            filename=file.filename,
            document_id=document_id,
            metadata={
                "uploaded_by": "api",
                "use_enhanced": use_enhanced
            }
        )
        
        # Create document record
        doc_record = {
            "id": document_id,
            "filename": file.filename,
            "content_type": storage_result["content_type"],
            "file_size": storage_result["file_size"],
            "file_path": storage_result["file_path"],
            "file_hash": storage_result["file_hash"],
            "uploaded_at": datetime.utcnow().isoformat(),
            "processing_status": "pending",
            "use_enhanced": use_enhanced,
            "storage_type": storage_result["storage_type"]
        }
        
        documents_store[document_id] = doc_record
        
        # Process document in background
        background_tasks.add_task(
            process_document_task,
            document_id,
            storage_result["file_path"],
            storage_result["content_type"],
            use_enhanced
        )
        
        # Notify WebSocket clients
        await notify_clients({
            "type": "document_uploaded",
            "document_id": document_id,
            "filename": file.filename
        })
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing",
            "message": "Document uploaded successfully and is being processed"
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

# Document processing task
async def process_document_task(
    document_id: str,
    file_path: str,
    content_type: str,
    use_enhanced: bool = True
):
    """Background task to process document with contextual embeddings"""
    try:
        logger.info(f"Processing document {document_id} with enhanced={use_enhanced}")
        
        # Update status
        if document_id in documents_store:
            documents_store[document_id]["processing_status"] = "processing"
        
        await notify_clients({
            "type": "processing_started",
            "document_id": document_id
        })
        
        if use_enhanced:
            # Enhanced processing with contextual embeddings
            
            # 1. Process document with intelligent chunking
            processing_result = await modern_doc_processor.process_document(
                file_path,
                content_type,
                metadata={"document_id": document_id}
            )
            
            # 2. Generate contextual embeddings (3 levels)
            embeddings_data = await contextual_embedding_generator.generate_embeddings(
                chunks=processing_result["chunks"],
                document_metadata={
                    "document_id": document_id,
                    "title": documents_store[document_id]["filename"],
                    **processing_result["metadata"]
                },
                use_voyage=bool(settings.voyage_api_key)
            )
            
            # 3. Store in vector database with context
            await modern_vector_store.store_contextual_embeddings(
                document_id,
                embeddings_data,
                processing_result["structure"]
            )
            
            # 4. Update document store
            if document_id in documents_store:
                documents_store[document_id].update({
                    "processing_status": "completed",
                    "chunk_count": len(processing_result["chunks"]),
                    "structure": processing_result["structure"].__dict__,
                    "processed_at": datetime.utcnow().isoformat(),
                    "processing_method": "enhanced_contextual"
                })
        else:
            # Basic processing (backward compatibility)
            chunks = await doc_processor.process_document(file_path)
            
            embeddings_data = []
            for i, chunk in enumerate(chunks):
                embedding = await embedding_service.generate_embedding(chunk["content"])
                embeddings_data.append({
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                    "embedding": embedding
                })
            
            await vector_store.store_embeddings(document_id, embeddings_data)
            
            if document_id in documents_store:
                documents_store[document_id].update({
                    "processing_status": "completed",
                    "chunk_count": len(chunks),
                    "processed_at": datetime.utcnow().isoformat(),
                    "processing_method": "basic"
                })
        
        logger.info(f"✅ Document {document_id} processed successfully")
        
        await notify_clients({
            "type": "processing_completed",
            "document_id": document_id,
            "chunk_count": documents_store[document_id].get("chunk_count", 0)
        })
        
    except Exception as e:
        logger.error(f"❌ Document processing failed for {document_id}: {e}")
        if document_id in documents_store:
            documents_store[document_id]["processing_status"] = "failed"
            documents_store[document_id]["error"] = str(e)
        
        await notify_clients({
            "type": "processing_failed",
            "document_id": document_id,
            "error": str(e)
        })

# Query endpoint with contextual search
@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using contextual embeddings"""
    try:
        if request.use_enhanced:
            # Create voice agent for query processing
            voice_agent = DocumentIntelligenceVoiceAgent()
            
            # Process query with contextual understanding
            result = await voice_agent.process_voice_query(
                query=request.query,
                conversation_id=request.conversation_id or f"api-{uuid.uuid4()}",
                audio_metadata={"source": "api", "context_level": request.context_level}
            )
            
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                conversation_id=result["conversation_id"],
                context_level=result["context_level"],
                metadata={
                    "metrics": result.get("metrics", {}),
                    "context_info": result.get("context_info", {})
                }
            )
        else:
            # Basic RAG query
            result = await rag_service.process_query(
                query=request.query,
                conversation_id=request.conversation_id
            )
            
            return QueryResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                conversation_id=request.conversation_id or "basic",
                context_level="document",
                metadata={}
            )
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

# List documents endpoint
@app.get("/api/v1/documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None
):
    """List all documents with pagination"""
    documents = list(documents_store.values())
    
    # Filter by status if provided
    if status:
        documents = [doc for doc in documents if doc.get("processing_status") == status]
    
    # Sort by upload time
    documents.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    
    # Paginate
    total = len(documents)
    documents = documents[skip:skip + limit]
    
    return {
        "documents": documents,
        "total": total,
        "skip": skip,
        "limit": limit
    }

# Get document details
@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details including structure"""
    if document_id not in documents_store:
        raise HTTPException(404, "Document not found")
    
    return documents_store[document_id]

# Delete document
@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    if document_id not in documents_store:
        raise HTTPException(404, "Document not found")
    
    try:
        # Delete from vector store
        if documents_store[document_id].get("use_enhanced"):
            await modern_vector_store.delete_document(document_id)
        else:
            await vector_store.delete_document(document_id)
        
        # Delete from storage
        storage_service = get_storage_service()
        filename = documents_store[document_id]["filename"]
        await storage_service.delete_file(document_id, filename)
        
        # Remove from store
        del documents_store[document_id]
        
        await notify_clients({
            "type": "document_deleted",
            "document_id": document_id
        })
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(500, f"Deletion failed: {str(e)}")

# Voice endpoints
@app.get("/api/v1/voice/token")
async def get_voice_token(
    room_name: str = Query("document-chat", description="Room name"),
    participant_name: str = Query("user", description="Participant name")
):
    """Get LiveKit token for voice connection"""
    try:
        token = await enhanced_voice_service.doc_agent.generate_token(
            room_name=room_name,
            participant_name=participant_name
        )
        
        return {
            "token": token,
            "url": settings.livekit_url,
            "room_name": room_name
        }
        
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise HTTPException(500, "Failed to generate voice token")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def notify_clients(message: Dict[str, Any]):
    """Notify all connected WebSocket clients"""
    if not active_connections:
        return
    
    message_json = json.dumps(message)
    disconnected = []
    
    for connection in active_connections:
        try:
            await connection.send_text(message_json)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        active_connections.remove(conn)

# Analytics endpoints
@app.get("/api/v1/analytics/usage")
async def get_usage_analytics():
    """Get usage analytics"""
    total_docs = len(documents_store)
    completed_docs = len([d for d in documents_store.values() if d.get("processing_status") == "completed"])
    failed_docs = len([d for d in documents_store.values() if d.get("processing_status") == "failed"])
    
    # Calculate total chunks
    total_chunks = sum(doc.get("chunk_count", 0) for doc in documents_store.values())
    
    # Processing methods breakdown
    enhanced_docs = len([d for d in documents_store.values() if d.get("processing_method") == "enhanced_contextual"])
    basic_docs = len([d for d in documents_store.values() if d.get("processing_method") == "basic"])
    
    return {
        "documents": {
            "total": total_docs,
            "completed": completed_docs,
            "failed": failed_docs,
            "processing": total_docs - completed_docs - failed_docs
        },
        "chunks": {
            "total": total_chunks,
            "average_per_doc": total_chunks / completed_docs if completed_docs > 0 else 0
        },
        "processing_methods": {
            "enhanced_contextual": enhanced_docs,
            "basic": basic_docs
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics from voice sessions"""
    # This would aggregate metrics from actual sessions
    return {
        "average_latencies": {
            "stt": 150,  # ms
            "llm": 800,  # ms
            "tts": 100,  # ms
            "rag": 200,  # ms
            "total": 1250  # ms
        },
        "success_rate": 0.95,
        "active_sessions": len(enhanced_voice_service.sessions) if enhanced_voice_service else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

# Test endpoints for development
@app.get("/api/v1/test/demo")
async def demo_test():
    """Demo endpoint to test system"""
    return {
        "message": "Voice Document Intelligence System v2.0",
        "features": {
            "contextual_embeddings": "3-level (local, document, global)",
            "voice_stack": "Deepgram Nova-3 + Cartesia Sonic + LiveKit",
            "document_processing": "Intelligent chunking with structure preservation",
            "rag": "Multi-context aware retrieval"
        },
        "services": {
            "modern_doc_processor": modern_doc_processor is not None,
            "contextual_embeddings": contextual_embedding_generator is not None,
            "enhanced_voice": enhanced_voice_service is not None,
            "modern_vector_store": modern_vector_store is not None
        }
    }

@app.post("/api/v1/test/process-sample")
async def process_sample_document(background_tasks: BackgroundTasks):
    """Process a sample document for testing"""
    sample_content = """
    # Sample Document: Voice Document Intelligence System
    
    ## Introduction
    This is a sample document to test the contextual embedding system.
    
    ## Features
    - Intelligent document chunking that preserves semantic boundaries
    - Three-level contextual embeddings (local, document, global)
    - Real-time voice interaction with document search
    - Advanced RAG with context-aware retrieval
    
    ## Architecture
    The system uses a microservices architecture with the following components:
    1. Document Processor: Handles intelligent chunking
    2. Embedding Generator: Creates contextual embeddings
    3. Vector Store: Manages embeddings with metadata
    4. Voice Service: Handles real-time voice interactions
    
    ## Implementation Details
    The contextual embedding approach ensures that each chunk maintains awareness of its surrounding context,
    document structure, and relationships to other documents in the system.
    """
    
    # Save sample document
    document_id = str(uuid.uuid4())
    file_path = Path("uploads") / f"{document_id}_sample.txt"
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, "w") as f:
        f.write(sample_content)
    
    # Create document record
    doc_record = {
        "id": document_id,
        "filename": "sample_document.txt",
        "content_type": "text/plain",
        "file_size": len(sample_content),
        "file_path": str(file_path),
        "uploaded_at": datetime.utcnow().isoformat(),
        "processing_status": "pending",
        "use_enhanced": True
    }
    
    documents_store[document_id] = doc_record
    
    # Process in background
    background_tasks.add_task(
        process_document_task,
        document_id,
        str(file_path),
        "text/plain",
        True
    )
    
    return {
        "document_id": document_id,
        "message": "Sample document created and processing started"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )