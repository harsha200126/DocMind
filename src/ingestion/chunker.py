# src/ingestion/chunker.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config


class DocumentChunker:
   

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Try to split on these separators first
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def chunk(self, documents: List[Document]) -> List[Document]:

        chunks = self.splitter.split_documents(documents)

        # Add chunk index to metadata for debugging and citation
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        print(f"Average chunk size: {sum(c.metadata['chunk_size'] for c in chunks) / len(chunks):.0f} chars")

        return chunks