-- infrastructure/local/init-db.sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema for better organization
CREATE SCHEMA IF NOT EXISTS doc_intel;

-- Documents table with advanced metadata
CREATE TABLE doc_intel.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(100),
    file_size BIGINT,
    file_hash VARCHAR(64), -- SHA-256 hash for deduplication
    
    -- Processing status with state machine
    processing_status VARCHAR(50) DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'queued', 'processing', 'completed', 'failed', 'archived')),
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    processing_error JSONB,
    
    -- Rich metadata
    metadata JSONB DEFAULT '{}',
    extracted_metadata JSONB DEFAULT '{}', -- Auto-extracted metadata
    user_metadata JSONB DEFAULT '{}', -- User-provided metadata
    
    -- Versioning
    version INTEGER DEFAULT 1,
    parent_document_id UUID REFERENCES doc_intel.documents(id),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE -- Soft delete
);

-- Document chunks with contextual information
CREATE TABLE doc_intel.document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES doc_intel.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64), -- For deduplication
    token_count INTEGER,
    
    -- Position information
    start_char INTEGER,
    end_char INTEGER,
    start_page INTEGER,
    end_page INTEGER,
    
    -- Structure information
    section_path TEXT[], -- e.g., ['Chapter 1', 'Section 1.2', 'Subsection 1.2.3']
    heading_level INTEGER,
    is_table BOOLEAN DEFAULT FALSE,
    is_list BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Embeddings (multiple models)
    embeddings JSONB DEFAULT '{}', -- Store multiple embeddings by model name
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Contextual embeddings with multi-level support
CREATE TABLE doc_intel.contextual_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES doc_intel.document_chunks(id) ON DELETE CASCADE,
    
    -- Context information
    context_type VARCHAR(50) NOT NULL CHECK (context_type IN ('local', 'document', 'global', 'semantic')),
    context_window INTEGER DEFAULT 3, -- Number of chunks used for context
    
    -- Model information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    embedding_dimension INTEGER NOT NULL,
    
    -- The actual embedding
    embedding vector(1536), -- Adjust dimension as needed
    
    -- Additional context data
    context_metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph entities extracted from documents
CREATE TABLE doc_intel.entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES doc_intel.documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES doc_intel.document_chunks(id) ON DELETE CASCADE,
    
    -- Entity information
    entity_type VARCHAR(100) NOT NULL, -- person, organization, concept, etc.
    entity_value TEXT NOT NULL,
    entity_normalized TEXT, -- Normalized/canonical form
    
    -- Confidence and source
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    extraction_method VARCHAR(50), -- NER model, regex, etc.
    
    -- Position in document
    start_offset INTEGER,
    end_offset INTEGER,
    
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Relationships between entities
CREATE TABLE doc_intel.entity_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES doc_intel.entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES doc_intel.entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User queries and interactions
CREATE TABLE doc_intel.queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_type VARCHAR(50), -- voice, text, semantic
    
    -- Query embeddings
    query_embedding vector(1536),
    enhanced_query_embedding vector(1536), -- After query enhancement
    
    -- Results
    results JSONB DEFAULT '[]',
    response_time_ms INTEGER,
    
    -- User feedback
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_feedback TEXT,
    
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_documents_status ON doc_intel.documents(processing_status) WHERE deleted_at IS NULL;
CREATE INDEX idx_documents_hash ON doc_intel.documents(file_hash) WHERE deleted_at IS NULL;
CREATE INDEX idx_documents_metadata ON doc_intel.documents USING gin(metadata);

CREATE INDEX idx_chunks_document ON doc_intel.document_chunks(document_id);
CREATE INDEX idx_chunks_content ON doc_intel.document_chunks USING gin(to_tsvector('english', content));

-- Vector similarity search indexes (using ivfflat)
CREATE INDEX idx_contextual_embeddings_vector ON doc_intel.contextual_embeddings 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_entities_type_value ON doc_intel.entities(entity_type, entity_value);
CREATE INDEX idx_entities_document ON doc_intel.entities(document_id);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION doc_intel.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON doc_intel.documents
    FOR EACH ROW EXECUTE FUNCTION doc_intel.update_updated_at_column();