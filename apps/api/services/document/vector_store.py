"""
Enhanced Vector Store Service combining original and modern multi-collection approach
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple

# Original imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Enhanced imports
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, Range, MatchValue, 
    SearchRequest, ScoredPoint, UpdateStatus, CollectionStatus, 
    OptimizersConfigDiff, HnswConfigDiff
)
import numpy as np

from apps.api.core.config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Original vector store service for backward compatibility"""
    
    def __init__(self):
        self.client = None
        self.collection_name = "documents"
        self.vector_size = 3072  # text-embedding-3-large dimension
        
    async def initialize(self):
        """Initialize Qdrant client and collection - original method preserved"""
        try:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30
            )
            
            # Create collection if it doesn't exist
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            raise
    
    async def store_embeddings(self, document_id: str, embeddings_data: List[Dict[str, Any]]):
        """Store embeddings in vector database - original method preserved"""
        try:
            if not self.client:
                await self.initialize()
            
            # Prepare points for insertion
            points = []
            for i, data in enumerate(embeddings_data):
                point = models.PointStruct(
                    id=f"{document_id}_{i}",
                    vector=data["embedding"],
                    payload={
                        "document_id": document_id,
                        "chunk_id": data["chunk_id"],
                        "content": data["content"],
                        "metadata": data.get("metadata", {}),
                        "chunk_index": i
                    }
                )
                points.append(point)
            
            # Insert points
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Embedding storage failed: {e}")
            raise
    
    async def search_similar(self, query_embedding: List[float], limit: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors - original method preserved"""
        try:
            if not self.client:
                await self.initialize()
            
            # Prepare filter if provided
            query_filter = None
            if filters:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filters.items()
                    ]
                )
            
            # Search
            search_results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "document_id": result.payload.get("document_id", ""),
                    "chunk_id": result.payload.get("chunk_id", "")
                })
            
            logger.debug(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def delete_document(self, document_id: str):
        """Delete all vectors for a document - original method preserved"""
        try:
            if not self.client:
                await self.initialize()
            
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted vectors for document {document_id}")
            
        except Exception as e:
            logger.error(f"Vector deletion failed: {e}")
            raise


class ModernVectorStore:
    """Enhanced vector store with multi-model and multi-collection support"""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30
        )
        
        self.collections = {
            "base": "document_chunks_base",
            "local": "document_chunks_local", 
            "document": "document_chunks_document",
            "global": "document_chunks_global",
            "semantic": "document_chunks_semantic"
        }
        
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure all required collections exist with optimal settings"""
        try:
            # Get existing collections
            existing = {c.name for c in self.client.get_collections().collections}
            
            # Collection settings for different embedding models
            vector_configs = {
                "openai": VectorParams(
                    size=3072,  # text-embedding-3-large dimension
                    distance=Distance.COSINE
                ),
                "voyage": VectorParams(
                    size=1024,  # voyage-3-large dimension
                    distance=Distance.COSINE
                ),
                "local": VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            }
            
            # Optimized HNSW configuration
            hnsw_config = HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            )
            
            optimizer_config = OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=5,
            )
            
            for context_type, collection_name in self.collections.items():
                if collection_name not in existing:
                    logger.info(f"Creating collection: {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vector_configs,  # Multi-vector support
                        hnsw_config=hnsw_config,
                        optimizers_config=optimizer_config,
                        on_disk_payload=True
                    )
        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            raise
    
    async def store_embeddings(
        self,
        document_id: str,
        embeddings_data: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Store multi-level embeddings in Qdrant"""
        stored_counts = {}
        
        for context_type in self.collections.keys():
            points = []
            
            for data in embeddings_data:
                point_id = str(uuid.uuid4())
                
                # Get embeddings for this context level
                context_embeddings = data["embeddings"].get(context_type, {})
                
                # Skip if no embeddings available
                if not any(context_embeddings.values()):
                    continue
                
                # Create payload
                payload = {
                    "document_id": document_id,
                    "chunk_index": data["chunk_index"],
                    "chunk_content": data["chunk_content"],
                    "context_type": context_type,
                    "context": data["contexts"].get(context_type, ""),
                    "metadata": data.get("metadata", {}),
                    "token_count": len(data["chunk_content"].split())
                }
                
                # Add section path if available
                if "section_path" in data.get("metadata", {}):
                    payload["section_path"] = data["metadata"]["section_path"]
                
                # Create point with multiple vectors
                point = PointStruct(
                    id=point_id,
                    vector=context_embeddings,  # Multi-vector support
                    payload=payload
                )
                points.append(point)
            
            # Store points in collection
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collections[context_type],
                        points=points
                    )
                    stored_counts[context_type] = len(points)
                    logger.info(f"Stored {len(points)} points in {self.collections[context_type]}")
                except Exception as e:
                    logger.error(f"Failed to store in {self.collections[context_type]}: {e}")
                    stored_counts[context_type] = 0
            else:
                stored_counts[context_type] = 0
        
        return stored_counts
    
    async def search(
        self,
        query_embeddings: Dict[str, List[float]],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        context_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search across multiple collections and embedding models"""
        if context_types is None:
            context_types = list(self.collections.keys())
        
        all_results = []
        
        # Prepare filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)
        
        # Search each context type
        for context_type in context_types:
            if context_type not in self.collections:
                continue
            
            collection_name = self.collections[context_type]
            
            # Search with multiple vector models
            for model_name, embedding in query_embeddings.items():
                if embedding is None:
                    continue
                
                try:
                    results = self.client.search(
                        collection_name=collection_name,
                        query_vector=(model_name, embedding),
                        query_filter=query_filter,
                        limit=top_k,
                        with_payload=True
                    )
                    
                    # Add context type and model info to results
                    for result in results:
                        all_results.append({
                            "id": result.id,
                            "score": result.score,
                            "payload": result.payload,
                            "context_type": context_type,
                            "model_used": model_name
                        })
                        
                except Exception as e:
                    logger.error(f"Search failed for {collection_name} with {model_name}: {e}")
        
        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    async def delete_document(self, document_id: str) -> Dict[str, int]:
        """Delete all chunks for a document from all collections"""
        deleted_counts = {}
        
        filter_cond = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
        
        for context_type, collection_name in self.collections.items():
            try:
                # Get count before deletion
                count_before = self.client.count(
                    collection_name=collection_name,
                    count_filter=filter_cond
                ).count
                
                # Delete points
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=filter_cond
                )
                
                deleted_counts[context_type] = count_before
                logger.info(f"Deleted {count_before} points from {collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to delete from {collection_name}: {e}")
                deleted_counts[context_type] = 0
        
        return deleted_counts