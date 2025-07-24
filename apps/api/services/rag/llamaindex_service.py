# apps/api/services/rag/llamaindex_service.py
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.tigerdb import TigerDBGraphStore
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.embeddings.voyage import VoyageEmbedding
from llama_index.llms.openai import OpenAI
import qdrant_client

class ModernRAGService:
    def __init__(self):
        # Initialize Qdrant for pure vector search
        self.qdrant_client = qdrant_client.QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True
        )
        
        # Initialize TigerGraph for hybrid search
        self.tiger_graph = TigerDBGraphStore(
            host="localhost",
            graph_name="document_intelligence",
            username="tigergraph",
            password="tigergraph"
        )
        
        # Use Voyage-3 embeddings (best commercial)
        self.embed_model = VoyageEmbedding(
            model_name="voyage-3-large",
            api_key=os.getenv("VOYAGE_API_KEY")
        )
        
        # Setup storage context
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="documents"
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            graph_store=self.tiger_graph
        )
        
    async def create_contextual_embeddings(self, documents):
        """Implement Anthropic's contextual retrieval approach"""
        # Add document context to chunks
        for doc in documents:
            doc.metadata["document_context"] = self._generate_context(doc)
        
        # Create index with contextual embeddings
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        return index