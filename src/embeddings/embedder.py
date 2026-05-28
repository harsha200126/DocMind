
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import config


class TextEmbedder:
  

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,        # "models/embedding-001"
            google_api_key=config.GOOGLE_API_KEY,
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string at search time.
        Returns a list of 768 floats.
        """
        return self.embeddings.embed_query(text)

    def get_embeddings_model(self):
        """
        Return the embeddings object for ChromaDB and LangChain.
        This is unchanged — ChromaDB calls this the same way regardless
        of whether it's OpenAI, Google, or any other provider.
        """
        return self.embeddings