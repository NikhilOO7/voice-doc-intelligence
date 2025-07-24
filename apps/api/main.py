# apps/api/main.py - Minimal version without Docker dependencies
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime

# Simple in-memory storage for demo
documents_store = {}
processing_queue = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting Voice Document Intelligence API (Demo Mode)")
    logger.info("📝 Note: Running without external dependencies for demo")
    yield
    logger.info("⚠️ Shutting down API")

# Create FastAPI app
app = FastAPI(
    title="Voice Document Intelligence API",
    version="1.0.0",
    description="Voice-enabled document intelligence system",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def simulate_processing(document_id: str):
    """Simulate document processing"""
    async def process():
        try:
            # Simulate processing time
            await asyncio.sleep(5)
            
            if document_id in documents_store:
                documents_store[document_id]['processing_status'] = 'completed'
                documents_store[document_id]['processing_completed_at'] = datetime.utcnow().isoformat()
                documents_store[document_id]['metadata'] = {
                    'chunks_count': 15,
                    'embeddings_stored': {'openai': 15, 'local': 15}
                }
                logger.info(f"✅ Document {document_id} processing completed")
            
        except Exception as e:
            logger.error(f"❌ Processing failed for {document_id}: {e}")
            if document_id in documents_store:
                documents_store[document_id]['processing_status'] = 'failed'
                documents_store[document_id]['processing_error'] = {'error': str(e)}
    
    # Run in background
    asyncio.create_task(process())

@app.get("/")
async def root():
    return {
        "name": "Voice Document Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "mode": "demo"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "storage": "demo_mode",
            "vector_db": "demo_mode",
            "voice": "demo_mode"
        }
    }

@app.post("/api/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
) -> Dict[str, Any]:
    """Upload a document for processing"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    max_size = 100 * 1024 * 1024  # 100MB
    if file.size and file.size > max_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Parse metadata
    user_metadata = {}
    if metadata:
        try:
            user_metadata = json.loads(metadata)
        except:
            pass
    
    try:
        # Generate IDs
        document_id = str(uuid.uuid4())
        external_id = str(uuid.uuid4())
        
        # Save file
        file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Store document info
        document_data = {
            "id": document_id,
            "external_id": external_id,
            "filename": file.filename,
            "file_type": file.content_type,
            "file_size": len(content),
            "processing_status": "queued",
            "processing_started_at": None,
            "processing_completed_at": None,
            "processing_error": None,
            "metadata": {},
            "extracted_metadata": {},
            "user_metadata": user_metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "file_path": str(file_path)
        }
        
        documents_store[document_id] = document_data
        
        # Start processing simulation
        logger.info(f"📄 Starting processing for {file.filename}")
        document_data["processing_status"] = "processing"
        document_data["processing_started_at"] = datetime.utcnow().isoformat()
        simulate_processing(document_id)
        
        return {
            "document_id": document_id,
            "external_id": external_id,
            "filename": file.filename,
            "status": "queued",
            "message": "Document queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str, include_chunks: bool = False) -> Dict[str, Any]:
    """Get document details"""
    
    # Try UUID first, then external_id
    document = documents_store.get(document_id)
    if not document:
        # Search by external_id
        for doc in documents_store.values():
            if doc.get("external_id") == document_id:
                document = doc
                break
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    response = {
        "id": document["id"],
        "external_id": document["external_id"],
        "filename": document["filename"],
        "file_type": document["file_type"],
        "file_size": document["file_size"],
        "processing_status": document["processing_status"],
        "metadata": document["metadata"],
        "extracted_metadata": document["extracted_metadata"],
        "user_metadata": document["user_metadata"],
        "created_at": document["created_at"],
        "updated_at": document["updated_at"]
    }
    
    if document["processing_error"]:
        response["error"] = document["processing_error"]
    
    if include_chunks and document["processing_status"] == "completed":
        # Simulate chunks
        response["chunks"] = [
            {
                "chunk_index": i,
                "content": f"Sample content chunk {i+1} from {document['filename']}...",
                "token_count": 150,
                "metadata": {"section": f"Section {i+1}"}
            }
            for i in range(3)  # Simulate 3 chunks
        ]
    
    return response

@app.get("/api/v1/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """List documents with pagination"""
    
    # Filter documents
    docs = list(documents_store.values())
    if status:
        docs = [doc for doc in docs if doc["processing_status"] == status]
    
    # Sort by creation time (newest first)
    docs.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    total = len(docs)
    paginated_docs = docs[skip:skip + limit]
    
    return {
        "documents": [
            {
                "id": doc["id"],
                "external_id": doc["external_id"],
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "file_size": doc["file_size"],
                "processing_status": doc["processing_status"],
                "created_at": doc["created_at"]
            }
            for doc in paginated_docs
        ],
        "pagination": {
            "total": total,
            "skip": skip,
            "limit": limit,
            "pages": (total + limit - 1) // limit if total > 0 else 1
        }
    }

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str) -> Dict[str, Any]:
    """Delete a document"""
    
    document = documents_store.get(document_id)
    if not document:
        # Search by external_id
        for doc_id, doc in documents_store.items():
            if doc.get("external_id") == document_id:
                document = doc
                document_id = doc_id
                break
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    try:
        file_path = Path(document["file_path"])
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete file: {e}")
    
    # Remove from store
    del documents_store[document_id]
    
    return {
        "message": "Document deleted successfully",
        "document_id": document_id
    }

# Voice endpoints (simplified demo versions)
@app.post("/api/v1/voice/create-room")
async def create_voice_room(
    room_name: Optional[str] = None,
    participant_name: str = "User"
) -> Dict[str, Any]:
    """Create a voice room (demo mode)"""
    
    if not room_name:
        room_name = f"doc-intelligence-{uuid.uuid4().hex[:8]}"
    
    # Return mock data for demo
    return {
        "room_name": room_name,
        "token": "demo_token_" + uuid.uuid4().hex[:16],
        "url": "ws://localhost:7880",  # This would be LiveKit URL
        "participant_name": participant_name,
        "room_info": {
            "name": room_name,
            "creation_time": datetime.utcnow().isoformat(),
            "max_participants": 5,
        },
        "demo_mode": True,
        "message": "Voice features require LiveKit setup"
    }

@app.get("/api/v1/voice/health")
async def voice_health_check() -> Dict[str, Any]:
    """Voice health check (demo mode)"""
    return {
        "status": "demo_mode",
        "livekit_configured": False,
        "openai_configured": False,
        "message": "Voice features available in full setup"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )