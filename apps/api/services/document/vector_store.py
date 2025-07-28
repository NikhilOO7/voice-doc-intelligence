# apps/api/services/document/vector_store.py
"""
Enhanced Vector Store Service with support for contextual embeddings
Manages multi-level context storage and retrieval
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

import qdrant_client
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams,
    CollectionStatus, OptimizersConfigDiff
)

from apps.api.core.config import settings
from apps.api.core.connections import get_qdrant_client, get_redis_client

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with contextual information"""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    local_context: Optional[Dict[str, Any]] = None
    document_context: Optional[Dict[str, Any]] = None
    global_context: Optional[Dict[str, Any]] = None

class VectorStoreService:
    """Original vector store service for backward compatibility"""
    
    def __init__(self):
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        
    async def initialize(self):
        """Initialize vector store"""
        client = await get_qdrant_client()
        
        # Create collection if it doesn't exist
        collections = await client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
    async def store_embeddings(
        self,
        document_id: str,
        embeddings_data: List[Dict[str, Any]]
    ):
        """Store embeddings in vector database"""
        client = await get_qdrant_client()
        
        points = []
        for data in embeddings_data:
            point = PointStruct(
                id=data["chunk_id"],
                vector=data["embedding"],
                payload={
                    "document_id": document_id,
                    "content": data["content"],
                    "metadata": data.get("metadata", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            points.append(point)
        
        # Upsert points
        await client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Stored {len(points)} embeddings for document {document_id}")
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        client = await get_qdrant_client()
        
        # Build filter
        qdrant_filter = None
        if filters and filters.get("document_ids"):
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=filters["document_ids"])
                    )
                ]
            )
        
        # Search
        results = await client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.id,
                "score": result.score,
                "document_id": result.payload.get("document_id"),
                "content": result.payload.get("content"),
                "metadata": result.payload.get("metadata", {})
            })
        
        return formatted_results
    
    async def delete_document(self, document_id: str):
        """Delete all embeddings for a document"""
        client = await get_qdrant_client()
        
        await client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        
        logger.info(f"Deleted embeddings for document {document_id}")

class ModernVectorStore:
    """Enhanced vector store with contextual embeddings support"""
    
    def __init__(self):
        # Collection names for different context levels
        self.collections = {
            "local": f"{settings.qdrant_collection_name}_local",
            "document": f"{settings.qdrant_collection_name}_document",
            "global": f"{settings.qdrant_collection_name}_global",
            "unified": f"{settings.qdrant_collection_name}_unified"
        }
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        self.cache_ttl = 3600  # 1 hour cache
        
    async def initialize(self):
        """Initialize all collections"""
        client = await get_qdrant_client()
        
        # Create collections for each context level
        existing_collections = [c.name for c in (await client.get_collections()).collections]
        
        for context_level, collection_name in self.collections.items():
            if collection_name not in existing_collections:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,
                        memmap_threshold=50000
                    )
                )
                logger.info(f"Created collection: {collection_name}")
    
    async def store_contextual_embeddings(
        self,
        document_id: str,
        embeddings_data: List[Dict[str, Any]],
        document_structure: Any
    ):
        """Store embeddings with contextual information"""
        client = await get_qdrant_client()
        redis_client = await get_redis_client()
        
        # Prepare points for each context level
        points_by_level = {
            "local": [],
            "document": [],
            "global": [],
            "unified": []
        }
        
        for idx, data in enumerate(embeddings_data):
            chunk_id = data["chunk_id"]
            base_payload = {
                "document_id": document_id,
                "chunk_index": idx,
                "content": data["content"],
                "metadata": data["metadata"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add structure information
            if document_structure:
                base_payload["document_structure"] = {
                    "title": document_structure.title,
                    "total_chunks": document_structure.total_chunks,
                    "document_type": document_structure.document_type,
                    "section": data["metadata"].get("section_path", [])
                }
            
            # Local context point
            local_payload = {**base_payload, "context_level": "local"}
            if "local_context" in data["metadata"]:
                local_payload["local_context"] = data["metadata"]["local_context"]
            
            points_by_level["local"].append(
                PointStruct(
                    id=f"{chunk_id}_local",
                    vector=data["embedding"],
                    payload=local_payload
                )
            )
            
            # Document context point
            doc_payload = {**base_payload, "context_level": "document"}
            if "document_context" in data["metadata"]:
                doc_payload["document_context"] = data["metadata"]["document_context"]
            
            points_by_level["document"].append(
                PointStruct(
                    id=f"{chunk_id}_document",
                    vector=data["embedding"],
                    payload=doc_payload
                )
            )
            
            # Global context point
            global_payload = {**base_payload, "context_level": "global"}
            if "global_context" in data["metadata"]:
                global_payload["global_context"] = data["metadata"]["global_context"]
            
            points_by_level["global"].append(
                PointStruct(
                    id=f"{chunk_id}_global",
                    vector=data["embedding"],
                    payload=global_payload
                )
            )
            
            # Unified point with all contexts
            unified_payload = {
                **base_payload,
                "context_level": "unified",
                "all_contexts": {
                    "local": data["metadata"].get("local_context"),
                    "document": data["metadata"].get("document_context"),
                    "global": data["metadata"].get("global_context")
                }
            }
            
            points_by_level["unified"].append(
                PointStruct(
                    id=f"{chunk_id}_unified",
                    vector=data["embedding"],
                    payload=unified_payload
                )
            )
        
        # Store points in respective collections
        for context_level, points in points_by_level.items():
            if points:
                await client.upsert(
                    collection_name=self.collections[context_level],
                    points=points
                )
                logger.info(f"Stored {len(points)} embeddings in {context_level} collection")
        
        # Cache document metadata in Redis
        if redis_client:
            cache_key = f"doc_meta:{document_id}"
            cache_data = {
                "structure": document_structure.__dict__ if document_structure else {},
                "chunk_count": len(embeddings_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            await redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
    
    async def search_with_context(
        self,
        query_embedding: List[float],
        context_level: str = "local",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search with specific context level"""
        client = await get_qdrant_client()
        
        # Determine collection to search
        if context_level == "auto":
            # Search all collections and merge results
            return await self._search_multi_context(query_embedding, filters, top_k)
        
        collection_name = self.collections.get(context_level, self.collections["unified"])
        
        # Build filter
        qdrant_filter = self._build_filter(filters)
        
        # Search
        results = await client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        search_results = []
        for result in results:
            payload = result.payload
            
            search_result = SearchResult(
                chunk_id=result.id,
                document_id=payload.get("document_id"),
                content=payload.get("content"),
                score=result.score,
                metadata=payload.get("metadata", {}),
                local_context=payload.get("local_context"),
                document_context=payload.get("document_context"),
                global_context=payload.get("global_context")
            )
            
            # Add document structure if available
            if "document_structure" in payload:
                search_result.metadata["document_structure"] = payload["document_structure"]
            
            search_results.append(search_result)
        
        return search_results
    
    async def _search_multi_context(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[SearchResult]:
        """Search across multiple context levels and merge results"""
        client = await get_qdrant_client()
        
        # Search each context level
        all_results = []
        context_weights = {
            "local": 0.4,
            "document": 0.35,
            "global": 0.25
        }
        
        for context_level, weight in context_weights.items():
            collection_name = self.collections[context_level]
            
            try:
                results = await client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    query_filter=self._build_filter(filters),
                    limit=top_k * 2,  # Get more results for merging
                    with_payload=True
                )
                
                # Weight scores by context level
                for result in results:
                    result.score *= weight
                    all_results.append((result, context_level))
                    
            except Exception as e:
                logger.error(f"Search failed for {context_level}: {e}")
        
        # Sort by weighted score and deduplicate
        all_results.sort(key=lambda x: x[0].score, reverse=True)
        
        seen_chunks = set()
        final_results = []
        
        for result, context_level in all_results:
            # Extract base chunk ID (without context suffix)
            base_chunk_id = result.id.rsplit('_', 1)[0]
            
            if base_chunk_id not in seen_chunks:
                seen_chunks.add(base_chunk_id)
                
                payload = result.payload
                search_result = SearchResult(
                    chunk_id=base_chunk_id,
                    document_id=payload.get("document_id"),
                    content=payload.get("content"),
                    score=result.score,
                    metadata={
                        **payload.get("metadata", {}),
                        "matched_context_level": context_level
                    },
                    local_context=payload.get("local_context"),
                    document_context=payload.get("document_context"),
                    global_context=payload.get("global_context")
                )
                
                final_results.append(search_result)
                
                if len(final_results) >= top_k:
                    break
        
        return final_results
    
    async def get_neighboring_chunks(
        self,
        chunk_id: str,
        window_size: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get neighboring chunks for local context"""
        client = await get_qdrant_client()
        redis_client = await get_redis_client()
        
        # Try cache first
        cache_key = f"neighbors:{chunk_id}:{window_size}"
        if redis_client:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Extract document ID and chunk index from chunk_id
        parts = chunk_id.split('_chunk_')
        if len(parts) != 2:
            return {"before": [], "after": []}
        
        document_id = parts[0]
        try:
            chunk_index = int(parts[1])
        except ValueError:
            return {"before": [], "after": []}
        
        # Get chunks before and after
        neighbors = {"before": [], "after": [], "section_path": []}
        
        # Search for neighboring chunks
        for offset in range(1, window_size + 1):
            # Before
            before_index = chunk_index - offset
            if before_index >= 0:
                before_chunk = await self._get_chunk_by_index(document_id, before_index)
                if before_chunk:
                    neighbors["before"].insert(0, before_chunk)
            
            # After
            after_index = chunk_index + offset
            after_chunk = await self._get_chunk_by_index(document_id, after_index)
            if after_chunk:
                neighbors["after"].append(after_chunk)
        
        # Get section path
        current_chunk = await self._get_chunk_by_index(document_id, chunk_index)
        if current_chunk and "metadata" in current_chunk:
            neighbors["section_path"] = current_chunk["metadata"].get("section_path", [])
        
        # Cache results
        if redis_client:
            await redis_client.setex(
                cache_key,
                300,  # 5 minutes
                json.dumps(neighbors)
            )
        
        return neighbors
    
    async def _get_chunk_by_index(
        self,
        document_id: str,
        chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by document ID and index"""
        client = await get_qdrant_client()
        
        # Search in unified collection
        results = await client.search(
            collection_name=self.collections["unified"],
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="chunk_index",
                        match=MatchValue(value=chunk_index)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if results:
            payload = results[0].payload
            return {
                "chunk_id": results[0].id,
                "content": payload.get("content"),
                "metadata": payload.get("metadata", {})
            }
        
        return None
    
    async def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata and structure"""
        redis_client = await get_redis_client()
        
        # Check cache
        if redis_client:
            cache_key = f"doc_meta:{document_id}"
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Query from vector store
        client = await get_qdrant_client()
        
        # Get first chunk to extract document structure
        results = await client.search(
            collection_name=self.collections["unified"],
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if results:
            payload = results[0].payload
            doc_structure = payload.get("document_structure", {})
            
            # Get all chunks to build complete structure
            all_chunks = await client.scroll(
                collection_name=self.collections["unified"],
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000
            )
            
            # Extract sections
            sections = set()
            for point in all_chunks[0]:
                section_path = point.payload.get("metadata", {}).get("section_path", [])
                if section_path:
                    sections.add(section_path[0])
            
            doc_info = {
                "title": doc_structure.get("title", "Unknown"),
                "summary": doc_structure.get("summary", ""),
                "sections": list(sections),
                "metadata": doc_structure,
                "chunk_count": len(all_chunks[0])
            }
            
            # Cache results
            if redis_client:
                await redis_client.setex(
                    f"doc_meta:{document_id}",
                    self.cache_ttl,
                    json.dumps(doc_info)
                )
            
            return doc_info
        
        return {}
    
    async def find_related_documents(
        self,
        document_id: str,
        query: str,
        max_related: int = 3
    ) -> List[Dict[str, Any]]:
        """Find documents related to the given document"""
        # This is a simplified implementation
        # In production, this would use a knowledge graph or citation analysis
        
        client = await get_qdrant_client()
        
        # Get sample embeddings from current document
        current_doc_chunks = await client.scroll(
            collection_name=self.collections["unified"],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=5,
            with_vectors=True
        )
        
        if not current_doc_chunks[0]:
            return []
        
        # Average embeddings
        embeddings = [point.vector for point in current_doc_chunks[0]]
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Search for similar documents, excluding current
        results = await client.search(
            collection_name=self.collections["global"],
            query_vector=avg_embedding,
            query_filter=Filter(
                must_not=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=max_related * 3  # Get more to ensure unique documents
        )
        
        # Aggregate by document
        related_docs = {}
        for result in results:
            doc_id = result.payload.get("document_id")
            if doc_id not in related_docs:
                related_docs[doc_id] = {
                    "document_id": doc_id,
                    "score": result.score,
                    "title": result.payload.get("document_structure", {}).get("title", "Unknown")
                }
        
        # Sort by score and return top N
        sorted_docs = sorted(
            related_docs.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_docs[:max_related]
    
    async def get_document_relationships(
        self,
        document_id: str
    ) -> Dict[str, List[Any]]:
        """Get relationships between documents"""
        # Simplified implementation
        # In production, this would use a graph database
        
        relationships = {
            "references": [],  # Documents referenced by this one
            "referenced_by": [],  # Documents that reference this one
            "entities": [],  # Shared entities
            "topics": []  # Shared topics
        }
        
        # Get document chunks to extract entities and topics
        client = await get_qdrant_client()
        
        chunks = await client.scroll(
            collection_name=self.collections["unified"],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=100,
            with_payload=True
        )
        
        # Extract entities and keywords
        all_entities = []
        all_keywords = []
        
        for point in chunks[0]:
            metadata = point.payload.get("metadata", {})
            all_entities.extend(metadata.get("entities", []))
            all_keywords.extend(metadata.get("keywords", []))
        
        # Deduplicate
        from collections import Counter
        
        entity_counts = Counter(e["text"] for e in all_entities if "text" in e)
        keyword_counts = Counter(all_keywords)
        
        relationships["entities"] = [
            {"text": text, "count": count}
            for text, count in entity_counts.most_common(10)
        ]
        
        relationships["topics"] = [
            {"topic": topic, "count": count}
            for topic, count in keyword_counts.most_common(10)
        ]
        
        return relationships
    
    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Build Qdrant filter from dictionary"""
        if not filters:
            return None
        
        conditions = []
        
        if "document_ids" in filters and filters["document_ids"]:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=filters["document_ids"])
                )
            )
        
        if "section" in filters:
            conditions.append(
                FieldCondition(
                    key="metadata.section_path",
                    match=MatchValue(value=filters["section"])
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    async def delete_document(self, document_id: str):
        """Delete all embeddings for a document across all collections"""
        client = await get_qdrant_client()
        redis_client = await get_redis_client()
        
        # Delete from all collections
        for context_level, collection_name in self.collections.items():
            try:
                await client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
                logger.info(f"Deleted embeddings from {collection_name} for document {document_id}")
            except Exception as e:
                logger.error(f"Failed to delete from {collection_name}: {e}")
        
        # Clear cache
        if redis_client:
            cache_keys = [
                f"doc_meta:{document_id}",
                f"neighbors:*{document_id}*"
            ]
            for pattern in cache_keys:
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)