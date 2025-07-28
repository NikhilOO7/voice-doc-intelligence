# apps/api/main_updated.py
"""
Enhanced FastAPI application with proper agent architecture
Integrates all document intelligence agents
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

# Agent imports - NEW ARCHITECTURE
from apps.api.services.agents import (
    DocumentIntelligenceCoordinator,
    AgentContext
)

# Storage services
from apps.api.services.storage.local_storage import LocalStorageService

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
coordinator: DocumentIntelligenceCoordinator = None
storage_service: LocalStorageService = None

# In-memory document store (replace with database in production)
documents_store: Dict[str, Dict] = {}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Voice Document Intelligence System with Agent Architecture...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("✅ Database initialized")
        
        # Initialize connections
        await init_redis()
        logger.info("✅ Redis initialized")
        
        await init_qdrant()
        logger.info("✅ Qdrant initialized")
        
        storage_service_instance = await init_storage()
        logger.info("✅ Storage initialized")
        
        # Initialize agent coordinator
        global coordinator, storage_service
        coordinator = DocumentIntelligenceCoordinator()
        storage_service = storage_service_instance
        logger.info("✅ Agent Coordinator initialized with all agents")
        
        # Log agent status
        agent_status = coordinator.get_agent_status()
        for agent_name, status in agent_status.items():
            logger.info(f"  - {agent_name.capitalize()} Agent: {'✅ Ready' if status['operational'] else '❌ Not Ready'}")
        
        logger.info("✅ Voice Document Intelligence System Ready!")
        logger.info(f"API: http://localhost:{settings.port}")
        logger.info(f"Web UI: http://localhost:{settings.port + 1}")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down Voice Document Intelligence System...")
        # Add cleanup code here
        logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Voice Document Intelligence API",
    description="AI-powered document intelligence with voice interaction and contextual embeddings",
    version="2.0.0",  # Updated version with agent architecture
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with agent status"""
    try:
        agent_status = coordinator.get_agent_status() if coordinator else {}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "agents": agent_status,
            "services": {
                "database": "connected",
                "redis": "connected",
                "qdrant": "connected",
                "storage": "ready"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Document endpoints
@app.post("/api/v1/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a document using Document Agent"""
    try:
        # Validate file
        if not file.filename.endswith(('.pdf', '.docx', '.txt', '.md')):
            raise HTTPException(400, "Unsupported file type")
        
        # Save file
        doc_id = str(uuid.uuid4())
        file_path = await storage_service.save_file(
            file.file,
            doc_id,
            file.filename,
            {"content_type": file.content_type}
        )
        
        # Process document with coordinator
        result = await coordinator.process_document(
            document_path=file_path,
            document_type=file.filename.split('.')[-1],
            user_id="default"  # Get from auth in production
        )
        
        if result["status"] == "failed":
            raise HTTPException(500, f"Document processing failed: {result.get('error')}")
        
        # Store document metadata
        document = Document(
            id=result["document_id"],
            title=file.filename,
            content_type=file.content_type,
            file_path=file_path,
            metadata=result["metadata"],
            chunk_count=result["chunks_created"],
            created_at=datetime.now()
        )
        
        # Add to database (simplified for now)
        documents_store[document.id] = document.dict()
        
        # Broadcast update
        await broadcast_update({
            "type": "document_uploaded",
            "document": document.dict()
        })
        
        return DocumentResponse(
            id=document.id,
            title=document.title,
            metadata=document.metadata,
            insights=result.get("insights", []),
            status="processed",
            created_at=document.created_at
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/v1/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List all documents"""
    try:
        # Get from in-memory store (replace with DB query)
        docs = list(documents_store.values())
        
        # Apply pagination
        paginated = docs[skip : skip + limit]
        
        return [
            DocumentResponse(
                id=doc["id"],
                title=doc["title"],
                metadata=doc["metadata"],
                status="processed",
                created_at=doc["created_at"]
            )
            for doc in paginated
        ]
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(500, str(e))

# Search endpoint
@app.post("/api/v1/search")
async def search_documents(
    query: str,
    context_levels: List[str] = Query(["local", "document"]),
    max_results: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Search documents using Query and Context Agents"""
    try:
        # Create agent context
        context = AgentContext(
            conversation_id=f"search_{uuid.uuid4()}",
            user_id="default"
        )
        
        # Process query through Query Agent
        query_result = await coordinator.query_agent.process({
            "query": query,
            "conversation_history": []
        }, context)
        
        # Search through Context Agent
        search_result = await coordinator.context_agent.process({
            "query": query_result["enhanced_query"],
            "context_levels": context_levels,
            "filters": {},
            "search_parameters": {
                **query_result["search_parameters"],
                "max_results": max_results
            }
        }, context)
        
        # Track search analytics
        await coordinator.analytics_agent.process({
            "action": "track",
            "event_data": {
                "type": "search",
                "query": query,
                "results_found": len(search_result["results"]),
                "context_levels": context_levels
            }
        }, context)
        
        return {
            "query": query,
            "enhanced_query": query_result["enhanced_query"],
            "intent": query_result["intent"],
            "results": search_result["results"],
            "total_results": search_result["search_metrics"]["total_candidates"],
            "context_graph": search_result.get("context_graph", {})
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(500, str(e))

# Voice endpoint
@app.websocket("/api/v1/voice/ws")
async def voice_websocket(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for real-time voice interaction"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Voice WebSocket connected: {session_id}")
        
        # Configure voice pipeline
        voice_config = await coordinator.voice_agent.process({
            "action": "configure",
            "room": None  # WebSocket mode, not LiveKit room
        }, AgentContext(session_id=session_id))
        
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process through coordinator
            result = await coordinator.process_voice_query(
                audio_data=data,
                session_id=session_id
            )
            
            # Send response
            await websocket.send_json({
                "transcript": result.get("transcript", ""),
                "response": result.get("response", ""),
                "search_results": result.get("search_results", []),
                "intent": result.get("intent", {}),
                "metrics": result.get("metrics", {})
            })
            
            # Stream audio if available
            if result.get("audio_stream"):
                async for audio_chunk in result["audio_stream"]:
                    await websocket.send_bytes(audio_chunk)
                    
    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
        await websocket.close()

# Analytics endpoints
@app.get("/api/v1/analytics/insights")
async def get_insights(
    time_range: str = Query("24h", regex="^\\d+[hdw]$"),
    db: AsyncSession = Depends(get_db)
):
    """Get system insights from Analytics Agent"""
    try:
        insights = await coordinator.generate_insights(time_range)
        return insights
        
    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/v1/analytics/metrics")
async def get_metrics():
    """Get agent performance metrics"""
    try:
        metrics = coordinator.get_agent_metrics()
        return {
            "agents": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(500, str(e))

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

async def broadcast_update(update: dict):
    """Broadcast update to all connected clients"""
    await manager.broadcast(update)

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main_updated:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )