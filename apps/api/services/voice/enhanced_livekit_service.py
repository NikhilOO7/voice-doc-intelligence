# apps/api/services/voice/enhanced_livekit_service.py
"""
Production-ready Voice Service with Deepgram Nova-3 STT + Cartesia Sonic TTS + LiveKit
Implements full DocumentIntelligenceVoiceAgent with 3-level contextual embeddings
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, AsyncIterator
from datetime import datetime
import numpy as np

# LiveKit imports
from livekit import api, rtc
from livekit.agents import (
    AutoSubscribe, JobContext, WorkerOptions,
    llm, stt, tts, voice
)
# OpenAI SDK
import openai

# Your app imports
from apps.api.core.config import settings
from apps.api.services.rag.llamaindex_service import ModernRAGService
from apps.api.services.document.vector_store import ModernVectorStore
from apps.api.services.document.embeddings import ContextualEmbeddingGenerator

logger = logging.getLogger(__name__)

# LiveKit plugin imports (after logger initialization)
try:
    from livekit.plugins import deepgram, cartesia, openai as lk_openai, silero
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False
    logger.warning("LiveKit plugins not available. Voice pipeline will be disabled.")

@dataclass
class ConversationContext:
    """Maintains conversation state and context"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_documents: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    last_query: Optional[str] = None
    conversation_start: datetime = field(default_factory=datetime.now)
    active_context_level: str = "local"  # local, document, global
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context_window(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context"""
        return self.messages[-max_messages:]
    
    def update_context_level(self, query: str):
        """Determine appropriate context level based on query"""
        if any(word in query.lower() for word in ["all", "everything", "global", "across"]):
            self.active_context_level = "global"
        elif any(word in query.lower() for word in ["document", "file", "report"]):
            self.active_context_level = "document"
        else:
            self.active_context_level = "local"

@dataclass
class LatencyMetrics:
    """Track performance metrics for optimization"""
    vad_latency: float = 0
    stt_latency: float = 0
    llm_latency: float = 0
    tts_latency: float = 0
    rag_latency: float = 0
    embedding_latency: float = 0
    total_latency: float = 0
    
    def log_metrics(self):
        """Log current metrics"""
        logger.info(f"Voice Pipeline Metrics - "
                   f"VAD: {self.vad_latency:.2f}ms, "
                   f"STT: {self.stt_latency:.2f}ms, "
                   f"LLM: {self.llm_latency:.2f}ms, "
                   f"TTS: {self.tts_latency:.2f}ms, "
                   f"RAG: {self.rag_latency:.2f}ms, "
                   f"Total: {self.total_latency:.2f}ms")

class DocumentIntelligenceVoiceAgent:
    """Enhanced voice agent with full contextual embedding support"""
    
    def __init__(self):
        self.rag_service = ModernRAGService()
        self.vector_store = ModernVectorStore()
        self.embedding_generator = ContextualEmbeddingGenerator()
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.metrics = LatencyMetrics()
        
    async def search_documents_with_context(
        self, 
        query: str,
        conversation_id: str,
        context_level: str = "local",
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Search documents using 3-level contextual embeddings"""
        try:
            start_time = time.time()
            
            # Get conversation context
            context = self.conversation_contexts.get(
                conversation_id, 
                ConversationContext()
            )
            
            # Generate query embedding with appropriate context
            query_embedding = await self.embedding_generator.generate_contextual_embedding(
                text=query,
                context_level=context_level,
                conversation_history=context.get_context_window(),
                document_context=context.current_documents
            )
            
            # Search with contextual embeddings
            search_results = await self.vector_store.search_with_context(
                query_embedding=query_embedding,
                context_level=context_level,
                filters={
                    "document_ids": context.current_documents if context_level == "document" else None
                },
                top_k=max_results
            )
            
            # Enhance results with multi-level context
            enhanced_results = []
            for result in search_results:
                # Get local context (surrounding chunks)
                local_context = await self._get_local_context(
                    chunk_id=result["chunk_id"],
                    window_size=2
                )
                
                # Get document context (document structure)
                doc_context = await self._get_document_context(
                    document_id=result["document_id"]
                )
                
                # Get global context (related documents)
                global_context = await self._get_global_context(
                    document_id=result["document_id"],
                    query=query
                )
                
                enhanced_results.append({
                    **result,
                    "local_context": local_context,
                    "document_context": doc_context,
                    "global_context": global_context,
                    "context_level": context_level
                })
            
            # Update metrics
            self.metrics.rag_latency = (time.time() - start_time) * 1000
            
            return {
                "results": enhanced_results,
                "context_level": context_level,
                "latency_ms": self.metrics.rag_latency
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
    
    async def _get_local_context(self, chunk_id: str, window_size: int = 2) -> Dict[str, Any]:
        """Get surrounding chunks for local context"""
        try:
            # Get neighboring chunks
            neighbors = await self.vector_store.get_neighboring_chunks(
                chunk_id=chunk_id,
                window_size=window_size
            )
            
            return {
                "preceding_chunks": neighbors.get("before", []),
                "following_chunks": neighbors.get("after", []),
                "section_path": neighbors.get("section_path", [])
            }
        except Exception as e:
            logger.error(f"Failed to get local context: {e}")
            return {}
    
    async def _get_document_context(self, document_id: str) -> Dict[str, Any]:
        """Get document structure and metadata"""
        try:
            # Get document metadata and structure
            doc_info = await self.vector_store.get_document_info(document_id)
            
            return {
                "title": doc_info.get("title", ""),
                "summary": doc_info.get("summary", ""),
                "sections": doc_info.get("sections", []),
                "metadata": doc_info.get("metadata", {}),
                "total_chunks": doc_info.get("chunk_count", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get document context: {e}")
            return {}
    
    async def _get_global_context(self, document_id: str, query: str) -> Dict[str, Any]:
        """Get related documents and cross-document relationships"""
        try:
            # Find related documents
            related_docs = await self.vector_store.find_related_documents(
                document_id=document_id,
                query=query,
                max_related=3
            )
            
            # Get cross-document entities and relationships
            relationships = await self.vector_store.get_document_relationships(
                document_id=document_id
            )
            
            return {
                "related_documents": related_docs,
                "cross_references": relationships.get("references", []),
                "shared_entities": relationships.get("entities", []),
                "topic_clusters": relationships.get("topics", [])
            }
        except Exception as e:
            logger.error(f"Failed to get global context: {e}")
            return {}
    
    async def process_voice_query(
        self,
        query: str,
        conversation_id: str,
        audio_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process voice query with full contextual understanding"""
        try:
            # Get or create conversation context
            if conversation_id not in self.conversation_contexts:
                self.conversation_contexts[conversation_id] = ConversationContext()
            
            context = self.conversation_contexts[conversation_id]
            
            # Update context level based on query
            context.update_context_level(query)
            
            # Add query to conversation
            context.add_message("user", query)
            context.last_query = query
            
            # Search with appropriate context level
            search_results = await self.search_documents_with_context(
                query=query,
                conversation_id=conversation_id,
                context_level=context.active_context_level
            )
            
            # Generate response using RAG with contextual embeddings
            response = await self.rag_service.generate_contextual_response(
                query=query,
                search_results=search_results["results"],
                conversation_context=context,
                audio_metadata=audio_metadata
            )
            
            # Add response to conversation
            context.add_message("assistant", response["answer"])
            
            # Extract action items and decisions
            if "action" in query.lower() or "todo" in query.lower():
                context.action_items.extend(response.get("action_items", []))
            if "decide" in query.lower() or "decision" in query.lower():
                context.decisions.extend(response.get("decisions", []))
            
            # Update current documents
            for result in search_results["results"]:
                doc_id = result["document_id"]
                if doc_id not in context.current_documents:
                    context.current_documents.append(doc_id)
            
            return {
                "answer": response["answer"],
                "sources": search_results["results"],
                "context_level": context.active_context_level,
                "conversation_id": conversation_id,
                "metrics": {
                    "rag_latency_ms": self.metrics.rag_latency,
                    "total_latency_ms": self.metrics.total_latency
                },
                "context_info": {
                    "action_items": context.action_items,
                    "decisions": context.decisions,
                    "documents_accessed": len(context.current_documents)
                }
            }
            
        except Exception as e:
            logger.error(f"Voice query processing failed: {e}")
            raise

class EnhancedVoiceService:
    """Production-ready voice service with Deepgram + Cartesia + LiveKit"""
    
    def __init__(self):
        self.doc_agent = DocumentIntelligenceVoiceAgent()
        self.sessions: Dict[str, Any] = {}
        
    async def create_voice_pipeline(self, room: rtc.Room):
        """Create optimized voice pipeline with Deepgram + Cartesia"""

        if not PLUGINS_AVAILABLE:
            raise RuntimeError(
                "Voice pipeline creation requires livekit plugin packages. "
                "Please install: livekit-plugins-deepgram, livekit-plugins-cartesia, livekit-plugins-silero"
            )

        logger.info("Creating voice pipeline with Deepgram STT + Cartesia TTS + Silero VAD")

        # Create Deepgram STT with Nova-3 (ultra-low latency)
        stt_plugin = deepgram.STT(
            model="nova-2-general",
            language="en-US",
            interim_results=True,
            punctuate=True,
            smart_format=True,
        )

        # Create Cartesia TTS with Sonic (ultra-low latency)
        tts_plugin = cartesia.TTS(
            model="sonic-english",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
            speed=1.0,
            emotion=["positivity:high", "curiosity:high"]
        )

        # Load Silero VAD for voice activity detection
        vad_plugin = silero.VAD.load()

        # Create custom LLM that integrates with document search
        custom_llm = DocumentAwareLLM(doc_agent=self.doc_agent)

        # Create initial chat context with system message
        initial_ctx = llm.ChatContext()
        initial_ctx.add_message(
            role="system",
            content="""You are an intelligent document assistant with voice capabilities.

You have access to a comprehensive document knowledge base with contextual embeddings.
Provide concise, helpful responses based on the documents available.
When answering, be conversational and natural."""
        )

        # Create voice agent with instructions
        agent = voice.Agent(
            instructions="""You are an intelligent document assistant with voice capabilities.

You have access to a comprehensive document knowledge base with contextual embeddings.
Provide concise, helpful responses based on the documents available.
When answering, be conversational and natural.""",
            vad=vad_plugin,
            stt=stt_plugin,
            llm=custom_llm,
            tts=tts_plugin,
            chat_ctx=initial_ctx,
        )

        # Start the agent
        agent.start(room)

        logger.info("Voice pipeline created and started successfully")
        return agent

class DocumentAwareLLM(llm.LLM):
    """Custom LLM that integrates with document search"""
    
    def __init__(self, doc_agent: DocumentIntelligenceVoiceAgent):
        super().__init__()
        self.doc_agent = doc_agent
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
    async def chat(
        self,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.ToolContext] = None,
        temperature: float = 0.7,
        n: int = 1,
        parallel_tool_calls: bool = True
    ) -> "llm.LLMStream":
        """Generate response with document context"""
        
        # Get last user message
        last_message = chat_ctx.messages[-1] if chat_ctx.messages else None
        if not last_message or last_message.role != "user":
            # No user message, use standard completion
            return await self._standard_completion(chat_ctx, temperature)
        
        # Search documents for context
        search_results = await self.doc_agent.search_documents_with_context(
            query=last_message.content,
            conversation_id="voice_session",
            context_level="auto"
        )
        
        # Inject document context into system message
        doc_context = self._format_search_results(search_results["results"])
        
        enhanced_messages = []
        
        # Add system message with document context
        system_msg = f"""You are an intelligent document assistant with access to the following relevant information:

{doc_context}

Use this information to provide accurate, helpful responses. If the information doesn't fully answer the question, acknowledge what you found and suggest how to get more complete information."""
        
        enhanced_messages.append({
            "role": "system",
            "content": system_msg
        })
        
        # Add conversation history
        for msg in chat_ctx.messages:
            enhanced_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Generate response
        response = await self._client.chat.completions.create(
            model="gpt-4",
            messages=enhanced_messages,
            temperature=temperature,
            stream=True,
            n=n
        )
        
        return DocumentLLMStream(response, self.doc_agent)
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for LLM context"""
        if not results:
            return "No relevant documents found."
        
        formatted = []
        for i, result in enumerate(results[:3]):  # Limit to top 3
            formatted.append(f"""
Document {i+1}: {result.get('document_title', 'Unknown')}
Section: {' > '.join(result.get('section_path', []))}
Content: {result.get('content', '')}
Relevance: {result.get('score', 0):.2f}
""")
        
        return "\n---\n".join(formatted)
    
    async def _standard_completion(self, chat_ctx: llm.ChatContext, temperature: float):
        """Fallback standard completion without document search"""
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_ctx.messages]
        
        response = await self._client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        return DocumentLLMStream(response, self.doc_agent)

class DocumentLLMStream(llm.LLMStream):
    """Custom LLM stream that tracks metrics"""
    
    def __init__(self, response_stream, doc_agent: DocumentIntelligenceVoiceAgent):
        self._stream = response_stream
        self._doc_agent = doc_agent
        self._start_time = time.time()
        
    async def __aiter__(self):
        async for chunk in self._stream:
            # Track first token latency
            if hasattr(self, '_start_time'):
                self._doc_agent.metrics.llm_latency = (time.time() - self._start_time) * 1000
                delattr(self, '_start_time')
            
            yield self._convert_chunk(chunk)
    
    def _convert_chunk(self, chunk):
        """Convert OpenAI chunk to LiveKit format"""
        # Implementation depends on LiveKit's expected format
        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(
                        role=chunk.choices[0].delta.role,
                        content=chunk.choices[0].delta.content
                    ),
                    index=0
                )
            ]
        )

# Entry point for LiveKit agent
async def entrypoint(ctx: JobContext):
    """Enhanced LiveKit agent entry point with full voice pipeline"""

    logger.info(f"Voice agent started for room: {ctx.room.name}")

    # Create voice service
    voice_service = EnhancedVoiceService()

    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create and start voice pipeline (agent is started inside create_voice_pipeline)
    agent = await voice_service.create_voice_pipeline(ctx.room)

    # Send initial greeting
    await agent.say(
        "Hello! I'm your document intelligence assistant. I can help you search through your documents, "
        "answer questions, and track important information from our conversation. What would you like to know?",
        allow_interruptions=True
    )
    
    # Log session start
    session_id = ctx.room.name
    voice_service.sessions[session_id] = {
        "start_time": datetime.now(),
        "agent": agent,
        "room": ctx.room
    }
    
    logger.info(f"Voice session started: {session_id}")

def create_worker_options() -> WorkerOptions:
    """Create worker options for voice agent"""
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        ws_url=settings.livekit_url,
        # worker_type removed - deprecated in newer versions
    )