"""
Enhanced RAG Service combining original and modern multi-agent approach
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
import time

# Original imports
from openai import OpenAI

# Enhanced imports  
from openai import AsyncOpenAI

from apps.api.core.config import settings
from apps.api.services.document.embeddings import EmbeddingService, ContextualEmbeddingGenerator
from apps.api.services.document.vector_store import VectorStoreService, ModernVectorStore

logger = logging.getLogger(__name__)

class RAGService:
    """Original RAG service for backward compatibility"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.conversation_history = {}  # In-memory for demo
        
    async def process_query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Original process_query method - preserved exactly"""
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = f"conv_{int(time.time())}"
            
            # Enhance query with conversation context
            enhanced_query = await self._enhance_query_with_context(query, conversation_id)
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(enhanced_query)
            
            # Search for relevant chunks
            search_results = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=5
            )
            
            # Generate answer using retrieved context
            answer = await self._generate_answer(enhanced_query, search_results)
            
            # Store conversation history
            self._update_conversation_history(conversation_id, query, answer)
            
            # Format sources
            sources = self._format_sources(search_results)
            
            return {
                "answer": answer,
                "sources": sources,
                "conversation_id": conversation_id,
                "query_enhanced": enhanced_query != query
            }
            
        except Exception as e:
            logger.error(f"RAG query processing failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "sources": [],
                "conversation_id": conversation_id or "error",
                "query_enhanced": False
            }
    
    async def _enhance_query_with_context(self, query: str, conversation_id: str) -> str:
        """Original query enhancement - preserved exactly"""
        try:
            # Get conversation history
            history = self.conversation_history.get(conversation_id, [])
            
            if not history:
                return query
            
            # Use AI to enhance query with context
            context_prompt = f"""
            Given this conversation history and new query, enhance the query for better document search:
            
            Conversation History:
            {json.dumps(history[-3:], indent=2)}  # Last 3 exchanges
            
            New Query: "{query}"
            
            Enhanced Query (add context but keep it concise):
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": context_prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            
            # Return enhanced query if it's different and reasonable
            if enhanced_query and len(enhanced_query) < len(query) * 2:
                return enhanced_query
            
            return query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query
    
    async def _generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Original answer generation - preserved exactly"""
        try:
            if not search_results:
                return "I couldn't find relevant information in the documents to answer your query."
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results):
                context_parts.append(f"Source {i+1}: {result['content']}")
            
            context_text = "\n\n".join(context_parts)
            
            # Generate answer using context
            system_prompt = """
            You are an intelligent document assistant. Use the provided context to answer the user's question accurately and helpfully.
            
            Guidelines:
            - Only use information from the provided context
            - If the context doesn't contain enough information, say so
            - Be concise but thorough
            - Cite specific sources when possible
            - If asked about something not in the context, clearly state that
            """
            
            user_prompt = f"""
            Context from documents:
            {context_text}
            
            Question: {query}
            
            Answer:
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I apologize, but I encountered an error generating the answer."
    
    def _update_conversation_history(self, conversation_id: str, query: str, answer: str):
        """Original conversation history update - preserved exactly"""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        self.conversation_history[conversation_id].append({
            "query": query,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Keep only last 10 exchanges per conversation
        if len(self.conversation_history[conversation_id]) > 10:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-10:]
    
    def _format_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Original source formatting - preserved exactly"""
        sources = []
        for result in search_results:
            sources.append({
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "similarity_score": result.get("score", 0),
                "metadata": result.get("metadata", {})
            })
        
        return sources


class ModernRAGService:
    """Enhanced RAG service with multi-agent and contextual processing"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_generator = ContextualEmbeddingGenerator()
        self.vector_store = ModernVectorStore()
        self.conversation_memory = {}
        
    async def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        context_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhanced query processing with multi-context search"""
        try:
            if not conversation_id:
                conversation_id = f"enhanced_conv_{int(time.time())}"
            
            # Analyze query intent
            query_analysis = await self._analyze_query_intent(query)
            
            # Generate contextual query embeddings
            query_embeddings = await self._generate_query_embeddings(query, query_analysis)
            
            # Multi-context search
            search_results = await self.vector_store.search(
                query_embeddings=query_embeddings,
                context_types=context_types,
                top_k=8
            )
            
            # Re-rank results based on query intent
            ranked_results = await self._rerank_results(search_results, query_analysis)
            
            # Generate comprehensive answer
            answer = await self._generate_enhanced_answer(
                query, query_analysis, ranked_results
            )
            
            # Update conversation memory
            self._update_conversation_memory(conversation_id, query, answer, query_analysis)
            
            return {
                "answer": answer,
                "sources": self._format_enhanced_sources(ranked_results),
                "conversation_id": conversation_id,
                "query_analysis": query_analysis,
                "enhanced_processing": True
            }
            
        except Exception as e:
            logger.error(f"Enhanced RAG processing failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error with enhanced processing: {str(e)}",
                "sources": [],
                "conversation_id": conversation_id or "error",
                "enhanced_processing": False
            }
    
    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and requirements"""
        try:
            analysis_prompt = f"""
            Analyze this query and provide structured information:
            
            Query: "{query}"
            
            Provide analysis in JSON format:
            {{
                "intent_type": "factual|procedural|analytical|comparison|summary",
                "key_entities": ["entity1", "entity2"],
                "query_complexity": "simple|moderate|complex",
                "context_requirements": ["local|document|global|semantic"],
                "expected_answer_length": "short|medium|long"
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except:
                return {
                    "intent_type": "factual",
                    "key_entities": [],
                    "query_complexity": "simple",
                    "context_requirements": ["document"],
                    "expected_answer_length": "medium"
                }
                
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"intent_type": "factual", "query_complexity": "simple"}
    
    async def _generate_query_embeddings(
        self, 
        query: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Generate embeddings optimized for query"""
        # Enhance query based on analysis
        enhanced_query = query
        
        if analysis.get("intent_type") == "procedural":
            enhanced_query = f"How to: {query}"
        elif analysis.get("intent_type") == "analytical":
            enhanced_query = f"Analysis of: {query}"
        
        # Generate embeddings using contextual generator
        embedding_result = await self.embedding_generator._generate_embedding(
            enhanced_query, use_voyage=True
        )
        
        return embedding_result
    
    async def _rerank_results(
        self, 
        results: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Re-rank results based on query analysis"""
        # Simple re-ranking based on context type preferences
        context_preferences = analysis.get("context_requirements", ["document"])
        
        def get_rank_score(result):
            base_score = result["score"]
            context_boost = 0.1 if result["context_type"] in context_preferences else 0
            return base_score + context_boost
        
        # Sort by enhanced score
        results.sort(key=get_rank_score, reverse=True)
        
        return results
    
    async def _generate_enhanced_answer(
        self,
        query: str,
        analysis: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> str:
        """Generate enhanced answer using analysis and results"""
        try:
            if not results:
                return "I couldn't find relevant information to answer your query."
            
            # Prepare enhanced context
            context_parts = []
            for i, result in enumerate(results[:5]):  # Top 5 results
                payload = result["payload"]
                context_type = result["context_type"]
                model_used = result["model_used"]
                
                context_parts.append(
                    f"[Context: {context_type}] {payload['chunk_content']}"
                )
            
            context_text = "\n\n".join(context_parts)
            
            # Enhanced system prompt based on analysis
            intent_type = analysis.get("intent_type", "factual")
            expected_length = analysis.get("expected_answer_length", "medium")
            
            system_prompt = f"""
            You are an advanced document intelligence assistant. Provide a {expected_length} {intent_type} response.
            
            Guidelines for {intent_type} queries:
            - Use the contextual information from multiple sources
            - Be comprehensive yet concise
            - Highlight key insights and connections
            - If analyzing, provide structured reasoning
            - If procedural, provide clear steps
            """
            
            user_prompt = f"""
            Multi-context information:
            {context_text}
            
            Query: {query}
            
            Enhanced Answer:
            """
            
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600 if expected_length == "long" else 400,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Enhanced answer generation failed: {e}")
            return "I apologize, but I encountered an error generating the enhanced answer."
    
    def _update_conversation_memory(
        self, 
        conversation_id: str, 
        query: str, 
        answer: str, 
        analysis: Dict[str, Any]
    ):
        """Update enhanced conversation memory"""
        if conversation_id not in self.conversation_memory:
            self.conversation_memory[conversation_id] = []
        
        self.conversation_memory[conversation_id].append({
            "query": query,
            "answer": answer,
            "analysis": analysis,
            "timestamp": time.time()
        })
        
        # Keep last 10 exchanges
        if len(self.conversation_memory[conversation_id]) > 10:
            self.conversation_memory[conversation_id] = self.conversation_memory[conversation_id][-10:]
    
    def _format_enhanced_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format enhanced source information"""
        sources = []
        for result in results:
            payload = result["payload"]
            sources.append({
                "document_id": payload.get("document_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "content": payload.get("chunk_content", "")[:200] + "..." if len(payload.get("chunk_content", "")) > 200 else payload.get("chunk_content", ""),
                "similarity_score": result.get("score", 0),
                "context_type": result.get("context_type", ""),
                "model_used": result.get("model_used", ""),
                "metadata": payload.get("metadata", {})
            })
        
        return sources