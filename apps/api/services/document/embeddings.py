"""
Enhanced Embedding Service combining original and new contextual embedding capabilities
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from functools import lru_cache

# Original imports
from openai import OpenAI

# Enhanced imports  
from openai import AsyncOpenAI
import voyageai
from sentence_transformers import SentenceTransformer
import torch

from apps.api.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Original embedding service for backward compatibility"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        
    async def generate_embedding(self, text: str, context: Optional[Dict] = None) -> List[float]:
        """Original generate_embedding method - preserved exactly"""
        try:
            # Enhance text with context if provided
            enhanced_text = self._enhance_with_context(text, context)
            
            # Generate embedding
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=enhanced_text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[str], contexts: Optional[List[Dict]] = None) -> List[List[float]]:
        """Original batch embedding generation - preserved exactly"""
        try:
            # Enhance texts with contexts
            enhanced_texts = []
            for i, text in enumerate(texts):
                context = contexts[i] if contexts and i < len(contexts) else None
                enhanced_texts.append(self._enhance_with_context(text, context))
            
            # Generate embeddings in batch
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=enhanced_texts
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
    
    def _enhance_with_context(self, text: str, context: Optional[Dict] = None) -> str:
        """Original context enhancement - preserved exactly"""
        if not context:
            return text
        
        # Add context prefixes to help with embedding quality
        enhanced_parts = []
        
        # Add document type context
        if context.get('document_type'):
            enhanced_parts.append(f"[Document: {context['document_type']}]")
        
        # Add section context
        if context.get('section'):
            enhanced_parts.append(f"[Section: {context['section']}]")
        
        # Add topic context
        if context.get('topic'):
            enhanced_parts.append(f"[Topic: {context['topic']}]")
        
        # Combine with original text
        if enhanced_parts:
            return " ".join(enhanced_parts) + " " + text
        
        return text


class ContextualEmbeddingGenerator:
    """Enhanced embedding generator with multi-model support"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Initialize Voyage AI client (better embeddings)
        if getattr(settings, 'voyage_api_key', None):
            self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
        else:
            self.voyage_client = None
        
        # Local model for fallback and comparison
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        except Exception as e:
            logger.warning(f"Local model initialization failed: {e}")
            self.local_model = None
        
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
        batch_start_idx: int,
        all_chunks: List[Dict[str, Any]],
        document_context: str,
        use_voyage: bool
    ) -> List[Dict[str, Any]]:
        """Process a batch of chunks"""
        batch_results = []
        
        for chunk_idx, chunk in enumerate(batch):
            absolute_idx = batch_start_idx + chunk_idx
            
            # Create multi-level contexts
            contexts = {
                "local": self._create_local_context(chunk, all_chunks, absolute_idx),
                "document": document_context,
                "global": await self._create_global_context(chunk, document_context),
                "semantic": await self._extract_semantic_context(chunk)
            }
            
            # Generate embeddings for each context level
            embeddings = {}
            for context_type, context_text in contexts.items():
                combined_text = f"{context_text} {chunk['content']}"
                embedding_result = await self._generate_embedding(combined_text, use_voyage)
                embeddings[context_type] = embedding_result
            
            # Create result
            batch_results.append({
                "chunk_index": chunk.get("chunk_index", absolute_idx),
                "chunk_content": chunk["content"],
                "embeddings": embeddings,
                "contexts": contexts,
                "metadata": chunk.get("metadata", {})
            })
        
        return batch_results
    
    def _create_document_context(self, metadata: Dict[str, Any]) -> str:
        """Create document-level context"""
        context_parts = []
        
        if metadata.get("title"):
            context_parts.append(f"Document: {metadata['title']}")
        
        if metadata.get("processing_method"):
            context_parts.append(f"Type: document")
        
        return " | ".join(context_parts)
    
    def _create_local_context(
        self, 
        chunk: Dict[str, Any], 
        all_chunks: List[Dict[str, Any]], 
        chunk_idx: int
    ) -> str:
        """Create local context from neighboring chunks"""
        context_parts = []
        
        # Previous chunk context
        if chunk_idx > 0:
            prev_content = all_chunks[chunk_idx - 1]["content"]
            context_parts.append(f"Previous: {prev_content[-100:]}")
        
        # Next chunk context
        if chunk_idx < len(all_chunks) - 1:
            next_content = all_chunks[chunk_idx + 1]["content"]
            context_parts.append(f"Next: {next_content[:100]}")
        
        return " | ".join(context_parts)
    
    async def _create_global_context(
        self, 
        chunk: Dict[str, Any], 
        document_context: str
    ) -> str:
        """Create global context"""
        # For now, use document context
        # In a full implementation, this would consider other documents
        return document_context
    
    async def _extract_semantic_context(self, chunk: Dict[str, Any]) -> str:
        """Extract semantic context using AI or keywords"""
        try:
            # Use OpenAI for semantic analysis
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 3-5 key concepts from this text. Return only the concepts separated by commas."
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
                    model=getattr(settings, 'voyage_model', 'voyage-3-large'),
                    input_type="document"
                )
                embeddings["voyage"] = result.embeddings[0]
            except Exception as e:
                logger.warning(f"Voyage embedding failed: {e}")
                embeddings["voyage"] = None
        
        # Always generate local embedding as fallback
        if self.local_model:
            try:
                local_embedding = self.local_model.encode(text, convert_to_tensor=True)
                embeddings["local"] = local_embedding.cpu().numpy().tolist()
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}")
                embeddings["local"] = None
        
        # Cache result
        self._embedding_cache[cache_key] = embeddings
        
        return embeddings