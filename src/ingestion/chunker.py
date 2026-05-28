# src/ingestion/chunker.py
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import config


class DocumentChunker:
    """
    Splits large documents into smaller overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in this order:
      1. Paragraphs (\n\n)
      2. Lines (\n)
      3. Sentences (. ! ?)
      4. Words (space)
      5. Characters (last resort)

    This means it respects the natural structure of text rather than
    blindly cutting at character 500. A sentence won't be split mid-word
    unless absolutely necessary.

    Enterprise alternatives:
    - SemanticChunker: uses embeddings to split at meaning boundaries
    - MarkdownTextSplitter: respects # headers and code blocks
    - HTMLSectionSplitter: splits on <h1>, <h2> tags
    - Custom splitters: for proprietary formats (legal docs, medical records)
    """

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
        """
        Split documents into chunks.

        Each chunk is still a Document object with:
        - page_content: the chunk text
        - metadata: inherited from parent (source file, page number)
                    PLUS chunk index added by us

        This metadata is how we tell users "this answer came from
        page 12 of policy_document.pdf"
        """
        chunks = self.splitter.split_documents(documents)

        # Add chunk index to metadata for debugging and citation
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        print(f"Average chunk size: {sum(c.metadata['chunk_size'] for c in chunks) / len(chunks):.0f} chars")

        return chunks