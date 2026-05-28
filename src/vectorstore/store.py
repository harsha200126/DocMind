# src/vectorstore/store.py
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from src.embeddings.embedder import TextEmbedder
from src.config import config


class VectorStore:
    """
    Stores embedded chunks and enables similarity search.

    ChromaDB is an open-source vector database that runs locally —
    perfect for development and small-to-medium production deployments.

    How it works internally:
    1. You give it a chunk of text
    2. It calls your embedding model to get a vector
    3. It stores both the text AND the vector
    4. At query time: embed the query → find nearest vectors → return text

    Enterprise alternatives (and when to use them):
    - Pinecone: fully managed, scales to billions of vectors,
                used by enterprises that don't want to run infrastructure
    - Weaviate: open-source, supports hybrid search (keyword + semantic)
    - Qdrant: open-source, fast, good Rust-based performance
    - pgvector: adds vector search to PostgreSQL — great if you're
                already on Postgres and don't want a new database
    - Azure AI Search / AWS OpenSearch: when you're in a cloud ecosystem
                and want managed infra + access controls + compliance
    """

    def __init__(self):
        self.embedder = TextEmbedder()
        self._store = None

    def create_from_documents(self, chunks: List[Document]) -> "VectorStore":
        """
        Embed all chunks and store them.
        This is the INGESTION step — run once when documents are added.
        """
        print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")
        self._store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedder.get_embeddings_model(),
            persist_directory=config.CHROMA_DB_PATH,
        )
        print(f"Vector store created at {config.CHROMA_DB_PATH}")
        return self

    def load_existing(self) -> "VectorStore":
        """
        Load a previously created vector store from disk.
        Production systems do this at startup instead of re-embedding
        everything on every restart — embedding is expensive.
        """
        self._store = Chroma(
            persist_directory=config.CHROMA_DB_PATH,
            embedding_function=self.embedder.get_embeddings_model(),
        )
        return self

    def as_retriever(self, top_k: int = config.TOP_K_RESULTS):
        """
        Returns a retriever object that LangChain chains can use.

        search_type="mmr" = Maximal Marginal Relevance
        Instead of returning the 4 most similar chunks (which might all
        say the same thing), MMR returns diverse chunks — the most
        relevant AND the least redundant. Better answers, less repetition.

        Enterprise enhancement: add metadata filters
        retriever = store.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"department": "legal"}  # Only legal docs
            }
        )
        """
        return self._store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 3},
        )