-- infrastructure/local/init-db.sql
-- Initialize database schema for Voice Document Intelligence

-- Create schema
CREATE SCHEMA IF NOT EXISTS doc_intel;

-- Set search path
SET search_path TO doc_intel, public;

-- Create extension for UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100),
    file_size BIGINT,
    file_path TEXT,
    file_hash VARCHAR(64),
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_method VARCHAR(50),
    processed_at TIMESTAMP WITH TIME ZONE,
    processing_error JSONB,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    token_count INTEGER,
    
    -- Position
    start_char INTEGER,
    end_char INTEGER,
    start_page INTEGER,
    end_page INTEGER,
    
    -- Structure
    section_path TEXT[],
    heading_level INTEGER,
    is_table BOOLEAN DEFAULT FALSE,
    is_list BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    embeddings JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(document_id, chunk_index)
);

-- Contextual embeddings table
CREATE TABLE IF NOT EXISTS contextual_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    
    -- Context levels
    context_level VARCHAR(20) NOT NULL CHECK (context_level IN ('local', 'document', 'global')),
    
    -- Embedding data
    embedding_model VARCHAR(100),
    embedding_dimension INTEGER,
    embedding_vector REAL[],
    
    -- Context data
    local_context JSONB,
    document_context JSONB,
    global_context JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(chunk_id, context_level)
);

-- Conversation sessions table
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    participant_name VARCHAR(100),
    
    -- Session data
    messages JSONB DEFAULT '[]',
    current_documents UUID[],
    action_items JSONB DEFAULT '[]',
    decisions JSONB DEFAULT '[]',
    
    -- Metrics
    metrics JSONB DEFAULT '{}',
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analytics events table
CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    session_id VARCHAR(255),
    document_id UUID REFERENCES documents(id),
    
    -- Event data
    event_data JSONB DEFAULT '{}',
    latency_ms INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_section ON document_chunks USING GIN(section_path);
CREATE INDEX idx_embeddings_chunk ON contextual_embeddings(chunk_id);
CREATE INDEX idx_embeddings_level ON contextual_embeddings(context_level);
CREATE INDEX idx_sessions_session_id ON conversation_sessions(session_id);
CREATE INDEX idx_analytics_type ON analytics_events(event_type);
CREATE INDEX idx_analytics_created ON analytics_events(created_at DESC);

-- Create update trigger for documents
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE
    ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();