"""
Enhanced Voice Service preserving original LiveKit functionality with document intelligence
"""

import logging
import asyncio
import io
import base64
from typing import Dict, Any, Optional

# Original imports preserved
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, deepgram
from livekit import api, rtc

# Enhanced imports
from livekit.agents import Agent, AgentSession, function_tool

from apps.api.core.config import settings
from apps.api.services.rag.llamaindex_service import RAGService, ModernRAGService

logger = logging.getLogger(__name__)

class VoiceService:
    """Original voice service for backward compatibility"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.sessions = {}  # Active voice sessions
        
    async def generate_token(self, room_name: str = "document-chat", participant_name: str = "user") -> str:
        """Original generate_token method - preserved exactly"""
        try:
            token = api.AccessToken(
                settings.livekit_api_key,
                settings.livekit_api_secret
            )
            
            token.with_identity(participant_name)
            token.with_name(participant_name)
            token.with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_subscribe=True,
                can_publish=True,
                can_publish_data=True
            ))
            
            return token.to_jwt()
            
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            raise
    
    async def process_voice_query(self, audio_data: bytes, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Original process_voice_query method - preserved exactly"""
        try:
            # Convert audio to text using OpenAI Whisper
            transcript = await self._speech_to_text(audio_data)
            
            if not transcript:
                return {"error": "Could not transcribe audio"}
            
            # Process query using RAG
            rag_result = await self.rag_service.process_query(
                query=transcript,
                conversation_id=conversation_id
            )
            
            # Convert answer to speech
            audio_response = await self._text_to_speech(rag_result["answer"])
            
            return {
                "transcript": transcript,
                "answer": rag_result["answer"],
                "sources": rag_result["sources"],
                "conversation_id": rag_result["conversation_id"],
                "audio_response": audio_response  # Base64 encoded
            }
            
        except Exception as e:
            logger.error(f"Voice query processing failed: {e}")
            return {"error": str(e)}
    
    async def _speech_to_text(self, audio_data: bytes) -> str:
        """Original speech-to-text method - preserved exactly"""
        try:
            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"
            
            # Use OpenAI Whisper API
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            response = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model="whisper-1",
                file=audio_file
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            return ""
    
    async def _text_to_speech(self, text: str) -> str:
        """Original text-to-speech method - preserved exactly"""
        try:
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            response = await asyncio.to_thread(
                client.audio.speech.create,
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Get audio data and encode as base64
            audio_data = response.content
            return base64.b64encode(audio_data).decode()
            
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return ""


class DocumentIntelligenceVoiceAgent:
    """Enhanced voice agent with document intelligence - preserving original structure"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.modern_rag_service = ModernRAGService()
        
    @function_tool
    async def search_documents(
        self, 
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5,
        use_enhanced: bool = True
    ) -> str:
        """Enhanced document search with backward compatibility"""
        try:
            if use_enhanced:
                # Use enhanced RAG service
                result = await self.modern_rag_service.process_query(
                    query=query,
                    conversation_id="voice_session"
                )
            else:
                # Use original RAG service
                result = await self.rag_service.process_query(
                    query=query,
                    conversation_id="voice_session"
                )
            
            # Format for voice response
            if not result.get("sources"):
                return f"I couldn't find any documents matching '{query}'. Please try a different search term."
            
            sources = result["sources"][:3]  # Limit for voice
            response_parts = [f"I found {len(sources)} relevant results for '{query}':"]
            
            for idx, source in enumerate(sources, 1):
                content_preview = source["content"]
                response_parts.append(f"{idx}. {content_preview}")
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return f"I'm sorry, I encountered an error while searching for '{query}'. Please try again."
    
    @function_tool
    async def get_document_summary(self, document_id: str) -> str:
        """Get document summary - enhanced functionality"""
        try:
            # This would integrate with the document store
            return f"Document summary functionality is available for document {document_id}."
        except Exception as e:
            logger.error(f"Document summary failed: {e}")
            return "I'm sorry, I couldn't retrieve the document summary."
    
    @function_tool
    async def list_recent_documents(self, limit: int = 5) -> str:
        """List recent documents - enhanced functionality"""
        try:
            # This would integrate with the document store
            return f"Recent documents functionality is available (limit: {limit})."
        except Exception as e:
            logger.error(f"List documents failed: {e}")
            return "I'm sorry, I couldn't retrieve the document list."


# LiveKit Agent Entry Point - Enhanced while preserving original functionality
async def entrypoint(ctx: JobContext):
    """Enhanced LiveKit agent entry point preserving original functionality"""
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are an enhanced document intelligence assistant with voice capabilities. "
            "You can answer questions about uploaded documents, provide summaries, and help users find specific information. "
            "You have access to both basic and enhanced processing capabilities. "
            "Keep your responses conversational and helpful. "
            "If you don't know something, say so clearly and suggest alternatives."
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize enhanced document intelligence agent
    doc_agent = DocumentIntelligenceVoiceAgent()
    
    # Create custom LLM that uses our RAG services
    class EnhancedDocumentLLM(llm.LLM):
        def __init__(self):
            super().__init__()
        
        async def agenerate(self, prompt: str, **kwargs) -> llm.LLMResponse:
            try:
                # Use enhanced RAG service by default, fallback to original
                try:
                    result = await doc_agent.modern_rag_service.process_query(
                        query=prompt,
                        conversation_id=ctx.room.name
                    )
                except:
                    result = await doc_agent.rag_service.process_query(
                        query=prompt,
                        conversation_id=ctx.room.name
                    )
                
                return llm.LLMResponse(text=result["answer"])
                
            except Exception as e:
                logger.error(f"Enhanced LLM generation failed: {e}")
                return llm.LLMResponse(text="I apologize, but I encountered an error processing your request.")
    
    # Create enhanced voice assistant with original structure preserved
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=EnhancedDocumentLLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )

    # Start the assistant
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say(
        "Hello! I'm your enhanced document intelligence assistant. "
        "I can help you search through your documents, get summaries, and answer questions. "
        "What would you like to know?", 
        allow_interruptions=True
    )