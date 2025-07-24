# apps/api/services/document/embeddings.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from openai import AsyncOpenAI
import voyageai
from sentence_transformers import SentenceTransformer
import torch
import logging
from functools import lru_cache

from ...core.config import settings
from ...core.connections import connections

logger = logging.getLogger(__name__)

class ContextualEmbeddingGenerator:
    """Generate multi-level contextual embeddings using multiple models"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Initialize Voyage AI client (better embeddings)
        if settings.voyage_api_key:
            self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
        else:
            self.voyage_client = None
        
        # Local model for fallback and comparison
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    async def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        document_metadata: Dict[str, Any],
        use_voyage: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate multi-level contextual embeddings"""
        embeddings_data = []
        
        # Create document context
        document_context = self._create_document_context(document_metadata)
        
        # Process chunks in batches for efficiency
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = await self._process_batch(
                batch, i, chunks, document_context, use_voyage
            )
            embeddings_data.extend(batch_embeddings)
        
        return embeddings_data
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        start_idx: int,
        all_chunks: List[Dict[str, Any]],
        document_context: str,
        use_voyage: bool
    ) -> List[Dict[str, Any]]:
        """Process a batch of chunks"""
        results = []
        
        for idx, chunk in enumerate(batch):
            chunk_idx = start_idx + idx
            
            # Create contexts
            local_context = self._create_local_context(all_chunks, chunk_idx)
            global_context = self._create_global_context(all_chunks, document_context)
            semantic_context = await self._create_semantic_context(chunk, all_chunks)
            
            # Generate embeddings for each context level
            embeddings = {}
            
            # Base embedding (just the chunk)
            embeddings["base"] = await self._generate_embedding(
                chunk["content"], use_voyage=use_voyage
            )
            
            # Local context embedding
            local_text = f"Local context: {local_context}\n\nContent: {chunk['content']}"
            embeddings["local"] = await self._generate_embedding(
                local_text, use_voyage=use_voyage
            )
            
            # Document context embedding
            doc_text = f"Document: {document_context}\n\nContent: {chunk['content']}"
            embeddings["document"] = await self._generate_embedding(
                doc_text, use_voyage=use_voyage
            )
            
            # Global context embedding
            global_text = f"Global context: {global_context}\n\nContent: {chunk['content']}"
            embeddings["global"] = await self._generate_embedding(
                global_text, use_voyage=use_voyage
            )
            
            # Semantic context embedding
            semantic_text = f"Related concepts: {semantic_context}\n\nContent: {chunk['content']}"
            embeddings["semantic"] = await self._generate_embedding(
                semantic_text, use_voyage=use_voyage
            )
            
            results.append({
                "chunk_index": chunk_idx,
                "chunk_content": chunk["content"],
                "embeddings": embeddings,
                "contexts": {
                    "local": local_context,
                    "document": document_context,
                    "global": global_context,
                    "semantic": semantic_context
                },
                "metadata": chunk.get("metadata", {})
            })
        
        return results
    
    def _create_local_context(
        self,
        chunks: List[Dict[str, Any]],
        current_idx: int,
        window_size: int = 2
    ) -> str:
        """Create context from surrounding chunks"""
        contexts = []
        
        # Previous chunks
        for i in range(max(0, current_idx - window_size), current_idx):
            preview = chunks[i]["content"][:150] + "..." if len(chunks[i]["content"]) > 150 else chunks[i]["content"]
            contexts.append(f"[Previous {current_idx - i}]: {preview}")
        
        # Next chunks
        for i in range(current_idx + 1, min(len(chunks), current_idx + window_size + 1)):
            preview = chunks[i]["content"][:150] + "..." if len(chunks[i]["content"]) > 150 else chunks[i]["content"]
            contexts.append(f"[Next {i - current_idx}]: {preview}")
        
        return "\n".join(contexts)
    
    def _create_document_context(self, metadata: Dict[str, Any]) -> str:
        """Create context from document metadata"""
        parts = [
            f"Document: {metadata.get('filename', 'Unknown')}",
            f"Type: {metadata.get('file_type', 'Unknown')}",
            f"Size: {metadata.get('file_size', 0):,} bytes"
        ]
        
        if "structure" in metadata:
            struct = metadata["structure"]
            parts.append(f"Structure: {struct.get('heading_count', 0)} headings, {struct.get('table_count', 0)} tables")
        
        if "extracted_metadata" in metadata:
            extracted = metadata["extracted_metadata"]
            if "title" in extracted:
                parts.insert(0, f"Title: {extracted['title']}")
            if "author" in extracted:
                parts.append(f"Author: {extracted['author']}")
        
        return " | ".join(parts)
    
    def _create_global_context(
        self,
        chunks: List[Dict[str, Any]],
        document_context: str
    ) -> str:
        """Create global context summary"""
        # Use first and last chunks for global overview
        parts = [document_context]
        
        if chunks:
            first_preview = chunks[0]["content"][:200] + "..." if len(chunks[0]["content"]) > 200 else chunks[0]["content"]
            parts.append(f"Document begins: {first_preview}")
            
            if len(chunks) > 1:
                last_preview = chunks[-1]["content"][:200] + "..." if len(chunks[-1]["content"]) > 200 else chunks[-1]["content"]
                parts.append(f"Document ends: {last_preview}")
        
        return " | ".join(parts)
    
    async def _create_semantic_context(
        self,
        chunk: Dict[str, Any],
        all_chunks: List[Dict[str, Any]]
    ) -> str:
        """Create semantic context using LLM to identify related concepts"""
        try:
            # Extract key concepts using LLM
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 3-5 key concepts or topics from the text. Return only the concepts separated by commas."
                    },
                    {
                        "role": "user",
                        "content": chunk["content"][:500]  # Limit context length
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            concepts = response.choices[0].message.content.strip()
            return concepts
            
        except Exception as e:
            logger.warning(f"Semantic context extraction failed: {e}")
            # Fallback to simple keyword extraction
            words = chunk["content"].split()
            # Get longest words as potential concepts
            concepts = sorted(set(w for w in words if len(w) > 6), key=len, reverse=True)[:5]
            return ", ".join(concepts)
    
    async def _generate_embedding(
        self,
        text: str,
        use_voyage: bool = True
    ) -> Dict[str, List[float]]:
        """Generate embeddings using multiple models"""
        # Check cache
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embeddings = {}
        
        # Generate OpenAI embedding
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=settings.embedding_model  # text-embedding-3-large
            )
            embeddings["openai"] = response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            embeddings["openai"] = None
        
        # Generate Voyage embedding (if available and requested)
        if use_voyage and self.voyage_client:
            try:
                result = self.voyage_client.embed(
                    texts=[text],
                    model=settings.voyage_model,  # voyage-3-large
                    input_type="document"
                )
                embeddings["voyage"] = result.embeddings[0]
            except Exception as e:
                logger.warning(f"Voyage embedding failed: {e}")
                embeddings["voyage"] = None
        
        # Always generate local embedding as fallback
        local_embedding = self.local_model.encode(text, convert_to_tensor=True)
        embeddings["local"] = local_embedding.cpu().numpy().tolist()
        
        # Cache the result
        self._embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    async def generate_query_embedding(
        self,
        query: str,
        enhance: bool = True
    ) -> Dict[str, Any]:
        """Generate embedding for search query with enhancement"""
        enhanced_query = query
        
        if enhance:
            enhanced_query = await self._enhance_query(query)
        
        # Generate embeddings with all models
        embeddings = await self._generate_embedding(enhanced_query, use_voyage=True)
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "embeddings": embeddings
        }
    
    async def _enhance_query(self, query: str) -> str:
        """Enhance query using LLM for better retrieval"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query enhancer. Add relevant synonyms, related terms, and clarifications to improve search results. Keep it concise."
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this search query: {query}"
                    }
                ],
                max_tokens=150,
                temperature=0.5
            )
            
            enhanced = response.choices[0].message.content.strip()
            return f"{query} {enhanced}"
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query