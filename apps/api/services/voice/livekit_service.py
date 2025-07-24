# apps/api/services/voice/livekit_service.py
import asyncio
import logging
from typing import AsyncIterable, Optional, Dict, Any
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool,
    Pipeline, VoicePipelineAgent
)
from livekit.plugins import openai, silero, deepgram, cartesia
from livekit import rtc
from livekit.agents import llm

from ...core.config import settings
from ...core.database import get_db_context
from ..rag.llamaindex_service import ModernRAGService
from ..document.vector_store import ModernVectorStore

logger = logging.getLogger(__name__)

class DocumentIntelligenceVoiceAgent:
    """Voice agent that integrates with document intelligence system"""
    
    def __init__(self):
        self.rag_service = ModernRAGService()
        self.vector_store = ModernVectorStore()
        
    @function_tool
    async def search_documents(
        self, 
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5
    ) -> str:
        """Search through uploaded documents for relevant information"""
        try:
            # Generate query embeddings
            from ..document.embeddings import ContextualEmbeddingGenerator
            embedding_gen = ContextualEmbeddingGenerator()
            
            query_embeddings = await embedding_gen.generate_query_embedding(
                query, enhance=True
            )
            
            # Search vector store
            filters = {}
            if document_type:
                filters["file_type"] = document_type
            
            results = await self.vector_store.search(
                query_embeddings["embeddings"],
                filters=filters,
                top_k=max_results
            )
            
            # Format results for voice response
            if not results:
                return f"I couldn't find any documents matching '{query}'. Please try a different search term."
            
            response_parts = [
                f"I found {len(results)} relevant results for '{query}':"
            ]
            
            for idx, result in enumerate(results[:3], 1):  # Limit to top 3 for voice
                payload = result["payload"]
                content_preview = payload["chunk_content"][:200] + "..." if len(payload["chunk_content"]) > 200 else payload["chunk_content"]
                
                response_parts.append(
                    f"{idx}. From document '{payload.get('document_name', 'Unknown')}': {content_preview}"
                )
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return f"I'm sorry, I encountered an error while searching for '{query}'. Please try again."
    
    @function_tool
    async def get_document_summary(self, document_id: str) -> str:
        """Get a summary of a specific document"""
        try:
            async with get_db_context() as db:
                from ...models.document import Document
                document = await db.get(Document, document_id)
                
                if not document:
                    return f"I couldn't find a document with ID {document_id}."
                
                if document.processing_status != "completed":
                    return f"Document '{document.filename}' is still being processed. Current status: {document.processing_status}"
                
                # Get metadata for summary
                metadata = document.metadata or {}
                extracted_metadata = document.extracted_metadata or {}
                
                summary_parts = [
                    f"Document: {document.filename}",
                    f"Type: {document.file_type}",
                    f"Size: {document.file_size:,} bytes",
                    f"Created: {document.created_at.strftime('%Y-%m-%d %H:%M')}"
                ]
                
                if metadata.get("chunks_count"):
                    summary_parts.append(f"Contains {metadata['chunks_count']} content sections")
                
                if extracted_metadata.get("title"):
                    summary_parts.append(f"Title: {extracted_metadata['title']}")
                
                return ". ".join(summary_parts)
                
        except Exception as e:
            logger.error(f"Document summary failed: {e}")
            return f"I'm sorry, I couldn't retrieve the summary for document {document_id}."
    
    @function_tool
    async def list_recent_documents(self, limit: int = 5) -> str:
        """List recently uploaded documents"""
        try:
            async with get_db_context() as db:
                from ...models.document import Document
                from sqlalchemy import select
                
                query = select(Document).where(
                    Document.deleted_at.is_(None)
                ).order_by(Document.updated_at.desc()).limit(limit)
                
                result = await db.execute(query)
                documents = result.scalars().all()
                
                if not documents:
                    return "No documents have been uploaded yet."
                
                response_parts = [f"Here are the {len(documents)} most recent documents:"]
                
                for idx, doc in enumerate(documents, 1):
                    status = "✓ Ready" if doc.processing_status == "completed" else f"⏳ {doc.processing_status.title()}"
                    response_parts.append(
                        f"{idx}. {doc.filename} ({status}) - uploaded {doc.created_at.strftime('%Y-%m-%d')}"
                    )
                
                return "\n".join(response_parts)
                
        except Exception as e:
            logger.error(f"List documents failed: {e}")
            return "I'm sorry, I couldn't retrieve the document list."

async def entrypoint(ctx: JobContext):
    """Main entry point for the voice agent"""
    await ctx.connect()
    
    # Initialize document intelligence voice agent
    doc_agent = DocumentIntelligenceVoiceAgent()
    
    # Configure the agent with document intelligence capabilities
    agent = Agent(
        instructions="""You are an intelligent document assistant with voice capabilities. 
        
        You help users interact with their uploaded documents through natural conversation. 
        You can:
        - Search through documents for specific information
        - Provide summaries of documents
        - List available documents
        - Answer questions about document content
        
        Always be helpful, concise, and conversational. When presenting search results, 
        summarize the key information rather than reading long passages verbatim.
        
        If a user asks about something not in the documents, politely let them know 
        and suggest they upload relevant documents first.""",
        
        tools=[
            doc_agent.search_documents,
            doc_agent.get_document_summary,
            doc_agent.list_recent_documents
        ],
    )
    
    # Configure the voice pipeline
    session = AgentSession(
        # Voice Activity Detection
        vad=silero.VAD.load(),
        
        # Speech to Text - using Deepgram for accuracy
        stt=deepgram.STT(
            model="nova-2",
            language="en",
            smart_format=True,
        ),
        
        # Large Language Model
        llm=openai.LLM(
            model=settings.openai_model,  # gpt-4-turbo
            temperature=0.7,
        ),
        
        # Text to Speech - using OpenAI for natural voice
        tts=openai.TTS(
            voice="alloy",
            speed=1.0,
        ),
        
        # Enable interruptions for natural conversation
        allow_interruptions=True,
        
        # Turn detection settings
        min_endpointing_delay=0.5,
        max_endpointing_delay=2.0,
    )
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and briefly explain your document intelligence capabilities. Keep it conversational and under 15 seconds."
    )

# Alternative Pipeline Agent Implementation (more control)
class DocumentVoicePipelineAgent(VoicePipelineAgent):
    """Custom pipeline agent with document intelligence integration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.doc_intelligence = DocumentIntelligenceVoiceAgent()
    
    async def on_enter(self):
        """Called when agent enters the room"""
        await self.session.say(
            "Hello! I'm your document intelligence assistant. I can help you search through your documents, get summaries, and answer questions about your uploaded content. What would you like to know?"
        )
    
    async def llm_node(self, chat_ctx: llm.ChatContext) -> AsyncIterable[llm.ChatChunk]:
        """Override LLM node to add document context when needed"""
        # Check if the user is asking about documents
        last_message = chat_ctx.messages[-1] if chat_ctx.messages else ""
        
        # Enhanced context for document-related queries
        if any(keyword in last_message.content.lower() 
               for keyword in ['document', 'search', 'find', 'file', 'pdf', 'summary']):
            
            # Add system context about available documents
            try:
                recent_docs = await self.doc_intelligence.list_recent_documents(3)
                enhanced_context = f"""
Current available documents context:
{recent_docs}

User query: {last_message.content}
                """
                
                # Create enhanced chat context
                enhanced_chat_ctx = chat_ctx.copy()
                enhanced_chat_ctx.messages[-1] = llm.ChatMessage(
                    role="user",
                    content=enhanced_context
                )
                
                # Use the enhanced context
                async for chunk in super().llm_node(enhanced_chat_ctx):
                    yield chunk
            except Exception as e:
                logger.error(f"Enhanced context failed: {e}")
                # Fall back to normal processing
                async for chunk in super().llm_node(chat_ctx):
                    yield chunk
        else:
            # Normal processing for non-document queries
            async for chunk in super().llm_node(chat_ctx):
                yield chunk

# Worker configuration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))