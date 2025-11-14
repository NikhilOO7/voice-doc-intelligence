# apps/api/services/voice/modern_voice_service.py
"""
MODERNIZED Voice Service - November 2025 Standards
Complete rewrite using latest SDK versions:
- Deepgram SDK v3+ (deepgram-sdk==5.3.0)
- Cartesia SDK v2.0+ (cartesia==2.0.15)
- LiveKit Agents SDK (livekit-agents==1.1.7)
- OpenAI SDK v1.x (openai==1.109.1)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

# LiveKit imports (modern patterns)
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    tokenize,
    tts,
    utils,
    vad,
    voice,
)
from livekit.plugins import deepgram, cartesia, openai as livekit_openai, silero

# Direct SDK imports for advanced features
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from cartesia import AsyncCartesia
from openai import AsyncOpenAI

# Your app imports
from apps.api.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Maintains conversation state across turns"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    document_refs: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def add_turn(self, role: str, content: str):
        """Add a conversation turn"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })


@dataclass
class VoiceMetrics:
    """Performance metrics for voice pipeline"""
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    rag_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "stt": self.stt_latency_ms,
            "llm": self.llm_latency_ms,
            "tts": self.tts_latency_ms,
            "rag": self.rag_latency_ms,
            "total": self.total_latency_ms,
        }


class ModernDeepgramSTT:
    """
    Deepgram SDK v3+ Speech-to-Text implementation
    Uses modern DeepgramClient with WebSocket streaming
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.deepgram_api_key
        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        self.client = DeepgramClient(self.api_key)
        self.model = "nova-2"  # Latest model as of Nov 2025
        logger.info(f"Initialized Deepgram STT with model: {self.model}")

    async def create_stream(self):
        """Create a Deepgram live transcription stream"""
        options = LiveOptions(
            model=self.model,
            language="en-US",
            smart_format=True,
            punctuate=True,
            profanity_filter=False,
            diarize=True,
            filler_words=False,
            utterance_end_ms=1000,
            vad_events=True,
            interim_results=True,
            endpointing=800,
        )

        connection = self.client.listen.websocket.v("1")

        # Set up event handlers
        @connection.on(LiveTranscriptionEvents.Transcript)
        async def on_transcript(result):
            sentence = result.channel.alternatives[0].transcript
            if sentence and result.is_final:
                logger.debug(f"Deepgram final transcript: {sentence}")

        @connection.on(LiveTranscriptionEvents.Error)
        async def on_error(error):
            logger.error(f"Deepgram error: {error}")

        await connection.start(options)
        return connection


class ModernCartesiaTTS:
    """
    Cartesia SDK v2.0+ Text-to-Speech implementation
    Uses AsyncCartesia with WebSocket streaming for ultra-low latency
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.cartesia_api_key
        if not self.api_key:
            raise ValueError("Cartesia API key is required")

        self.client = AsyncCartesia(api_key=self.api_key)
        self.voice_id = settings.cartesia_voice_id
        self.model = "sonic-english"  # Latest model as of Nov 2025
        logger.info(f"Initialized Cartesia TTS with model: {self.model}")

    async def synthesize_stream(self, text: str):
        """
        Synthesize speech using WebSocket streaming for lowest latency

        Modern Cartesia v2.0 API supports:
        - client.tts.sse() for Server-Sent Events
        - client.tts.bytes() for simple byte streaming
        - client.tts.websocket() for bidirectional WebSocket (best for real-time)
        """
        try:
            # Use SSE streaming for reliable async iteration
            async for chunk in self.client.tts.sse(
                model_id=self.model,
                transcript=text,
                voice_id=self.voice_id,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 24000,
                },
                language="en",
                _experimental_voice_controls={
                    "speed": "normal",
                    "emotion": ["neutral"],
                },
            ):
                if chunk.audio:
                    yield chunk.audio
        except Exception as e:
            logger.error(f"Cartesia TTS error: {e}")
            raise


class DocumentRAGIntegration:
    """
    Integration with RAG service for document-aware conversations
    """

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4-turbo-preview"  # Latest model

    async def query_with_context(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        document_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Query LLM with document context using modern OpenAI SDK v1.x

        Modern patterns:
        - AsyncOpenAI client (not global functions)
        - Streaming with async for
        - Proper error handling
        """
        messages = []

        # System prompt
        messages.append({
            "role": "system",
            "content": """You are an intelligent document assistant with real-time voice capabilities.

Your role:
- Answer questions based on document context when available
- Keep responses concise and conversational for voice output
- Cite specific documents when referencing information
- Ask clarifying questions if needed

Guidelines:
- Use natural, conversational language
- Keep answers to 1-2 sentences unless details are requested
- Always acknowledge when you don't have information"""
        })

        # Add document context if available
        if document_context:
            context_text = "\n\n".join([
                f"Document: {doc.get('title', 'Untitled')}\n{doc.get('content', '')[:500]}"
                for doc in document_context[:3]  # Top 3 docs
            ])
            messages.append({
                "role": "system",
                "content": f"Relevant document context:\n{context_text}"
            })

        # Add conversation history (last 10 turns)
        for msg in conversation_history[-10:]:
            messages.append(msg)

        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })

        # Stream response using modern v1.x API
        try:
            response_text = ""
            stream = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,  # Reasonable for voice
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

            return response_text

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I'm having trouble processing that request. Could you try again?"


class ModernVoiceAssistant:
    """
    Production-ready Voice Assistant using LiveKit Agents SDK

    Implements the modern pattern with proper plugin architecture
    """

    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.metrics = VoiceMetrics()
        self.rag_integration = DocumentRAGIntegration()

        # Initialize plugins using LiveKit's plugin system
        self.stt_plugin = None  # Will be initialized per-session
        self.tts_plugin = None  # Will be initialized per-session

        logger.info("Modern Voice Assistant initialized")

    async def handle_voice_session(self, ctx: JobContext):
        """
        Handle a LiveKit voice session

        Modern LiveKit Agents pattern:
        1. Use plugins (deepgram, cartesia, etc.)
        2. Create VoiceAssistant instance
        3. Handle events properly
        """
        room_name = ctx.room.name

        # Initialize conversation state
        if room_name not in self.conversations:
            self.conversations[room_name] = ConversationState()

        conversation = self.conversations[room_name]

        logger.info(f"Starting voice session for room: {room_name}")

        # Connect to room with auto-subscribe to audio only
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Initialize STT using LiveKit plugin (modern pattern)
        stt_instance = deepgram.STT(
            api_key=settings.deepgram_api_key,
            model="nova-2",
            language="en-US",
        )

        # Initialize TTS using LiveKit plugin (modern pattern)
        tts_instance = cartesia.TTS(
            api_key=settings.cartesia_api_key,
            voice_id=settings.cartesia_voice_id,
        )

        # Create LLM function that integrates RAG
        async def llm_function(chat_ctx: llm.ChatContext):
            """Custom LLM function with RAG integration"""
            # Extract user message
            user_message = chat_ctx.messages[-1].content if chat_ctx.messages else ""

            # TODO: Query document store for relevant context
            # For now, use empty context
            document_context = None

            # Get response from RAG-enhanced LLM
            response = await self.rag_integration.query_with_context(
                query=user_message,
                conversation_history=[
                    {"role": m.role, "content": m.content}
                    for m in chat_ctx.messages
                ],
                document_context=document_context
            )

            # Add to conversation state
            conversation.add_turn("assistant", response)

            # Return as ChatChunk for streaming
            return llm.ChatChunk(
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            content=response
                        )
                    )
                ]
            )

        # Create Voice Assistant using modern SDK
        assistant = voice.VoiceAssistant(
            vad=silero.VAD.load(),
            stt=stt_instance,
            llm=livekit_openai.LLM(model="gpt-4-turbo"),  # Use plugin LLM
            tts=tts_instance,
            chat_ctx=llm.ChatContext(
                messages=[
                    llm.ChatMessage(
                        role="system",
                        content="You are a helpful voice assistant with access to documents."
                    )
                ]
            ),
        )

        # Start assistant
        assistant.start(ctx.room)

        # Send greeting
        await asyncio.sleep(0.5)
        await assistant.say(
            "Hello! I'm your AI assistant. How can I help you today?",
            allow_interruptions=True
        )

        # Set up event handlers
        @assistant.on("user_speech_committed")
        def on_user_speech(msg: llm.ChatMessage):
            logger.info(f"User: {msg.content}")
            conversation.add_turn("user", msg.content)

        @assistant.on("agent_speech_committed")
        def on_agent_speech(msg: llm.ChatMessage):
            logger.info(f"Assistant: {msg.content}")

        # Handle room events
        @ctx.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"Participant joined: {participant.identity}")

        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(f"Participant left: {participant.identity}")

        # Keep session alive
        await asyncio.Event().wait()


# Worker entrypoint for LiveKit Agents
async def entrypoint(ctx: JobContext):
    """
    Modern LiveKit worker entrypoint

    This function is called by the LiveKit Agents framework
    for each new voice session/room
    """
    assistant = ModernVoiceAssistant()
    await assistant.handle_voice_session(ctx)


def create_worker():
    """
    Create and configure the LiveKit worker

    Run this worker with:
    python -m livekit.agents.cli dev modern_voice_service.py
    """
    return cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            port=8081,
        )
    )


if __name__ == "__main__":
    # For local development/testing
    create_worker()
