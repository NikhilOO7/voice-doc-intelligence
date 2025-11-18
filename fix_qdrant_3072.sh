#!/bin/bash
# Fix Qdrant collection dimensions for 3072 (text-embedding-3-large)

echo "🔧 Fixing Qdrant collections for text-embedding-3-large (3072 dims)..."

# Delete old collections
for collection in voice_doc_embeddings_local voice_doc_embeddings_document voice_doc_embeddings_global voice_doc_embeddings_unified; do
  curl -X DELETE http://localhost:6333/collections/$collection 2>/dev/null
  echo "Deleted $collection"
done

echo ""
echo "Creating new collections with 3072 dimensions..."

# Create all collections
for collection in voice_doc_embeddings_local voice_doc_embeddings_document voice_doc_embeddings_global voice_doc_embeddings_unified; do
  curl -X PUT http://localhost:6333/collections/$collection \
    -H "Content-Type: application/json" \
    -d '{
      "vectors": {
        "size": 3072,
        "distance": "Cosine"
      }
    }' 2>/dev/null
  echo "✅ Created $collection"
done

echo ""
echo "🎉 All collections recreated with 3072 dimensions!"
