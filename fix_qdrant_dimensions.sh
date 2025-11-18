#!/bin/bash
# Fix Qdrant collection dimensions for 1024 (text-embedding-ada-002)

echo "🔧 Fixing Qdrant collection dimensions..."

# Delete old collections
echo "Deleting old collections..."
curl -X DELETE http://localhost:6333/collections/voice_doc_embeddings_local 2>/dev/null
curl -X DELETE http://localhost:6333/collections/voice_doc_embeddings_document 2>/dev/null
curl -X DELETE http://localhost:6333/collections/voice_doc_embeddings_global 2>/dev/null

echo ""
echo "Creating new collections with 1024 dimensions..."

# Create local collection
curl -X PUT http://localhost:6333/collections/voice_doc_embeddings_local \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1024,
      "distance": "Cosine"
    }
  }' 2>/dev/null

echo "✅ Created voice_doc_embeddings_local"

# Create document collection
curl -X PUT http://localhost:6333/collections/voice_doc_embeddings_document \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1024,
      "distance": "Cosine"
    }
  }' 2>/dev/null

echo "✅ Created voice_doc_embeddings_document"

# Create global collection
curl -X PUT http://localhost:6333/collections/voice_doc_embeddings_global \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1024,
      "distance": "Cosine"
    }
  }' 2>/dev/null

echo "✅ Created voice_doc_embeddings_global"

echo ""
echo "🎉 All Qdrant collections recreated with 1024 dimensions!"
echo "You can now upload documents successfully."

