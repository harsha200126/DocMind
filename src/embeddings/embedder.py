# src/embeddings/embedder.py
from typing import List
from langchain_openai import OpenAIEmbeddings
from src.config import config


class TextEmbedder:


    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string — used at search time."""
        return self.embeddings.embed_query(text)

    def get_embeddings_model(self):

        return self.embeddings