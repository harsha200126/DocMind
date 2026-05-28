from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.vectorstore.store import VectorStore
from src.llm.chain import RAGChain
import os

def ingest_documents(doc_dir: str = "./data/documents"):
    """
    One-time setup: load documents → chunk → embed → store.
    """
    print("=== INGESTION PIPELINE ===")

    loader = DocumentLoader()
    chunker = DocumentChunker()
    store = VectorStore()

    documents = loader.load_directory(doc_dir)
    chunks = chunker.chunk(documents)
    store.create_from_documents(chunks)

    print("Ingestion complete. Vector store is ready.")
    return store # <-- We return the open database here

def load_and_query(active_store=None):
    """
    Normal operation: load existing vector store → answer questions.
    """
    # If we just created the store, use it. Otherwise, load from disk.
    if active_store is None:
        store = VectorStore().load_existing()
    else:
        store = active_store

    retriever = store.as_retriever()
    chain = RAGChain(retriever)

    print("\n=== DocMind is ready. Type 'quit' to exit. ===\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = chain.ask(question)
        print(f"\nDocMind: {result['answer']}")
        print("\nSources:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['file']} (page {src['page']})")
            print(f"       \"{src['excerpt']}\"")
        print()

if __name__ == "__main__":
# Force the pipeline to run and embed the documents
    current_store = ingest_documents()
    
    # Load the chat interface using the freshly populated database
    load_and_query(current_store)