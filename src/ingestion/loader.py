from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)


class DocumentLoader:


    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF file."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents

    def load_text(self, file_path: str) -> List[Document]:
        """Load a plain text file."""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def load_directory(self, dir_path: str) -> List[Document]:
        """
        Load ALL supported files from a folder.
        Enterprise use: point this at an S3 bucket, SharePoint,
        or Google Drive — same interface, different loader.
        """
        documents = []
        path = Path(dir_path)

        # Load all PDFs
        pdf_loader = DirectoryLoader(
            dir_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents.extend(pdf_loader.load())

        # Load all text files
        txt_loader = DirectoryLoader(
            dir_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents.extend(txt_loader.load())

        print(f"Total documents loaded: {len(documents)}")
        return documents