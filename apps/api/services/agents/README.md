# Document Intelligence Agents

This module implements the agent-based architecture for the Voice-Enabled Document Intelligence System as specified in the POC implementation plan.

## Overview

The system uses five specialized agents that work together to provide comprehensive document intelligence capabilities:

### 1. Document Agent 📄
**Role**: Document Intelligence Specialist  
**Responsibilities**:
- Document ingestion and intelligent chunking
- Multi-level context extraction (local, document, global)
- Metadata enrichment and tagging
- Document relationship mapping
- Structure preservation and hierarchy understanding

**Key Features**:
- Intelligent chunking that preserves semantic boundaries
- Three-level contextual embeddings
- Rich metadata extraction using GPT-4
- Document relationship mapping

### 2. Voice Agent 🎤
**Role**: Voice Conversation Specialist  
**Responsibilities**:
- Real-time speech-to-text with Deepgram Nova-3
- Natural text-to-speech with Cartesia Sonic
- Voice activity detection (VAD)
- Audio stream management
- Natural conversation flow control

**Key Features**:
- Ultra-low latency voice processing
- Natural turn-taking management
- Emotion-aware TTS
- WebSocket and LiveKit support

### 3. Query Agent 🔍
**Role**: Query Understanding Specialist  
**Responsibilities**:
- Intent recognition and classification
- Query enhancement and expansion
- Contextual memory management
- Query reformulation for better search
- Conversation state tracking

**Key Features**:
- Advanced intent analysis using GPT-4
- Contextual reference resolution
- Query variation generation
- Entity extraction

### 4. Context Agent 🧠
**Role**: Contextual Search Specialist  
**Responsibilities**:
- Contextual embeddings search (local, document, global)
- Hybrid search strategies (semantic + keyword + metadata)
- Multi-factor ranking and relevance scoring
- Context expansion and related content discovery
- Hierarchical retrieval respecting document structure

**Key Features**:
- Multi-level contextual search
- Hybrid scoring algorithm
- Smart result ranking
- Context graph generation

### 5. Analytics Agent 📊
**Role**: Business Intelligence Analyst  
**Responsibilities**:
- Usage tracking and pattern analysis
- Knowledge gap identification
- Performance monitoring and optimization
- Predictive analytics for content needs
- Proactive insight generation

**Key Features**:
- Real-time usage analytics
- Pattern detection using clustering
- Knowledge gap analysis
- Predictive modeling

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Document Intelligence Coordinator           │
│                    (Orchestrates all agents)            │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │                                         │
┌───▼─────┐  ┌────────────┐  ┌─────────┐  ┌─▼───────┐  ┌───────────┐
│Document │  │   Voice    │  │  Query  │  │Context │  │Analytics │
│ Agent   │  │   Agent    │  │  Agent  │  │ Agent  │  │  Agent   │
└─────────┘  └────────────┘  └─────────┘  └─────────┘  └───────────┘
```

## Usage

### Basic Usage

```python
from apps.api.services.agents import DocumentIntelligenceCoordinator, AgentContext

# Initialize coordinator
coordinator = DocumentIntelligenceCoordinator()

# Create context
context = AgentContext(
    conversation_id="conv_123",
    session_id="session_456",
    user_id="user_789"
)

# Process a document
result = await coordinator.process_document(
    document_path="/path/to/document.pdf",
    document_type="pdf",
    user_id="user_789"
)

# Process a voice query
voice_result = await coordinator.process_voice_query(
    audio_data=audio_bytes,
    session_id="session_456"
)

# Generate insights
insights = await coordinator.generate_insights(
    time_range="24h",
    focus_areas=["usage_patterns", "knowledge_gaps"]
)
```

### Individual Agent Usage

```python
# Use Document Agent directly
doc_agent = coordinator.document_agent
doc_result = await doc_agent.process({
    "document_path": "/path/to/doc.pdf",
    "document_type": "pdf"
}, context)

# Use Query Agent directly
query_agent = coordinator.query_agent
query_result = await query_agent.process({
    "query": "What are the safety protocols?",
    "conversation_history": []
}, context)
```

### CrewAI Integration

```python
# Run a complex task with a crew
crew_result = await coordinator.run_crew_task(
    crew_name="document_analysis",
    task_description="Analyze quarterly report for key insights",
    context={"document_id": "doc_123"}
)
```

## Configuration

Each agent can be configured through environment variables:

```bash
# Voice Agent
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key
CARTESIA_VOICE_ID=sonic-english

# All Agents
OPENAI_API_KEY=your_openai_key

# Analytics
ANALYTICS_MIN_PATTERN_FREQUENCY=3
ANALYTICS_PERFORMANCE_THRESHOLD=2000
```

## Workflows

### Voice Query Workflow
1. Voice Agent: Speech → Text
2. Query Agent: Enhance query & detect intent
3. Context Agent: Search documents
4. Voice Agent: Response → Speech

### Document Processing Workflow
1. Document Agent: Process & chunk document
2. Analytics Agent: Track processing metrics

### Insight Generation Workflow
1. Analytics Agent: Analyze usage patterns
2. Context Agent: Assess coverage
3. Query Agent: Identify gaps

## Metrics and Monitoring

Each agent tracks its own metrics:

```python
# Get agent metrics
metrics = coordinator.get_agent_metrics()

# Example output:
{
    "document_agent": {
        "operations": 156,
        "average_latency_ms": 342.5,
        "errors": 2
    },
    "voice_agent": {
        "operations": 523,
        "average_latency_ms": 125.3,
        "errors": 0
    }
}
```

## Error Handling

All agents implement comprehensive error handling:

```python
try:
    result = await agent.process(input_data, context)
except Exception as e:
    # Errors are logged and metrics updated
    # Graceful fallbacks are provided
    logger.error(f"Agent error: {e}")
```

## Testing

```bash
# Run agent tests
pytest apps/api/services/agents/tests/

# Test individual agents
pytest apps/api/services/agents/tests/test_document_agent.py
```

## Future Enhancements

1. **Multi-language Support**: Extend agents to handle multiple languages
2. **Advanced Analytics**: Add ML models for deeper insights
3. **Agent Learning**: Implement feedback loops for agent improvement
4. **Distributed Processing**: Scale agents across multiple workers
5. **Custom Agent Creation**: Framework for creating domain-specific agents