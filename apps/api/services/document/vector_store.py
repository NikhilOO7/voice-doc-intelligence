# apps/api/services/document/vector_store.py
from typing import List, Dict, Any, Optional, Tuple
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    Range, MatchValue, SearchRequest, ScoredPoint, UpdateStatus,
    CollectionStatus, OptimizersConfigDiff, HnswConfigDiff
)
import numpy as np
import logging

from ...core.connections import connections
from ...core.config import settings

logger = logging.getLogger(__name__)

class ModernVectorStore:
    """Advanced vector store using Qdrant with multi-model support"""
    
    def __init__(self):
        self.client = connections.get_qdrant()
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
        
        # Optimized HNSW configuration for better search performance
        hnsw_config = HnswConfigDiff(
            m=16,  # Number of connections
            ef_construct=100,  # Construction time accuracy
            full_scan_threshold=10000,  # When to switch to exact search
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
                    on_disk_payload=True  # Store payload on disk for large documents
                )
    
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
                    vector=context_embeddings,  # Dict of model_name: embedding
                    payload=payload
                )
                points.append(point)
            
            # Batch upload to Qdrant
            if points:
                self.client.upsert(
                    collection_name=self.collections[context_type],
                    points=points,
                    wait=True
                )
                stored_counts[context_type] = len(points)
                logger.info(f"Stored {len(points)} points in {self.collections[context_type]}")
        
        return stored_counts
    
    async def search(
        self,
        query_embeddings: Dict[str, List[float]],
        context_weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Advanced search across multiple context levels"""
        
        if context_weights is None:
            context_weights = {
                "base": 0.3,
                "local": 0.2,
                "document": 0.2,
                "global": 0.15,
                "semantic": 0.15
            }
        
        # Build filter conditions
        filter_conditions = []
        if filters:
            if "document_id" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=filters["document_id"])
                    )
                )
            if "min_tokens" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="token_count",
                        range=Range(gte=filters["min_tokens"])
                    )
                )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search in each context collection
        all_results = []
        
        for context_type, weight in context_weights.items():
            if weight <= 0:
                continue
            
            collection_name = self.collections[context_type]
            
            # Search with the best available embedding
            vector_name = None
            query_vector = None
            
            # Prefer Voyage > OpenAI > Local
            if "voyage" in query_embeddings and query_embeddings["voyage"]:
                vector_name = "voyage"
                query_vector = query_embeddings["voyage"]
            elif "openai" in query_embeddings and query_embeddings["openai"]:
                vector_name = "openai"
                query_vector = query_embeddings["openai"]
            elif "local" in query_embeddings and query_embeddings["local"]:
                vector_name = "local"
                query_vector = query_embeddings["local"]
            
            if not query_vector:
                continue
            
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=(vector_name, query_vector),
                    limit=top_k,
                    query_filter=search_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False  # Don't return vectors to save bandwidth
                )
                
                # Add context weight to results
                for result in results:
                    all_results.append({
                        "id": result.id,
                        "score": result.score * weight,
                        "base_score": result.score,
                        "context_type": context_type,
                        "weight": weight,
                        "payload": result.payload
                    })
                    
            except Exception as e:
                logger.error(f"Search failed in {collection_name}: {e}")
        
        # Combine and rank results
        combined_results = self._combine_results(all_results)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results[:top_k]
    
    def _combine_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results from multiple context searches"""
        # Group by document_id and chunk_index
        result_groups = {}
        
        for result in all_results:
            payload = result["payload"]
            key = f"{payload['document_id']}_{payload['chunk_index']}"
            
            if key not in result_groups:
                result_groups[key] = {
                    "document_id": payload["document_id"],
                    "chunk_index": payload["chunk_index"],
                    "chunk_content": payload["chunk_content"],
                    "metadata": payload.get("metadata", {}),
                    "scores": {},
                    "contexts": {},
                    "combined_score": 0.0
                }
            
            # Add score and context for this search
            result_groups[key]["scores"][result["context_type"]] = {
                "score": result["base_score"],
                "weighted_score": result["score"]
            }
            result_groups[key]["contexts"][result["context_type"]] = payload.get("context", "")
            result_groups[key]["combined_score"] += result["score"]
        
        return list(result_groups.values())
    
    async def get_similar_chunks(
        self,
        document_id: str,
        chunk_index: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar chunks within the same document"""
        # First, get the chunk's embedding
        chunk_filter = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index))
            ]
        )
        
        # Get the chunk from base collection
        results = self.client.scroll(
            collection_name=self.collections["base"],
            scroll_filter=chunk_filter,
            limit=1,
            with_vectors=True
        )
        
        if not results[0]:
            return []
        
        chunk = results[0][0]
        
        # Search for similar chunks in the same document
        similar_filter = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ],
            must_not=[
                FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index))
            ]
        )
        
        # Use the chunk's vector to find similar chunks
        similar_results = []
        for vector_name, vector in chunk.vector.items():
            if vector:
                results = self.client.search(
                    collection_name=self.collections["base"],
                    query_vector=(vector_name, vector),
                    limit=top_k,
                    query_filter=similar_filter,
                    with_payload=True
                )
                similar_results.extend(results)
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for result in sorted(similar_results, key=lambda x: x.score, reverse=True):
            chunk_idx = result.payload["chunk_index"]
            if chunk_idx not in seen:
                seen.add(chunk_idx)
                unique_results.append({
                    "chunk_index": chunk_idx,
                    "similarity_score": result.score,
                    "content": result.payload["chunk_content"],
                    "metadata": result.payload.get("metadata", {})
                })
        
        return unique_results[:top_k]
    
    async def update_chunk_metadata(
        self,
        document_id: str,
        chunk_index: int,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific chunk across all collections"""
        success = True
        
        for collection_name in self.collections.values():
            try:
                # Find points matching document_id and chunk_index
                filter_cond = Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                        FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index))
                    ]
                )
                
                # Get matching points
                results = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_cond,
                    limit=10,
                    with_payload=True
                )
                
                # Update each point's metadata
                for point in results[0]:
                    updated_payload = point.payload.copy()
                    updated_payload["metadata"].update(metadata_updates)
                    
                    self.client.set_payload(
                        collection_name=collection_name,
                        payload=updated_payload,
                        points=[point.id]
                    )
                    
            except Exception as e:
                logger.error(f"Failed to update metadata in {collection_name}: {e}")
                success = False
        
        return success
    
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