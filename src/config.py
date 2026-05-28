import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file into environment variables

class Config:

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/embedding-001"

    # Chunking settings (we'll explain these deeply in Step 6)
    CHUNK_SIZE: int = 500       # Characters per chunk
    CHUNK_OVERLAP: int = 50     # Overlap between chunks

    # Retrieval settings
    TOP_K_RESULTS: int = 4      # How many chunks to retrieve per query

    # Vector store
    CHROMA_DB_PATH: str = "./chroma_db"

config = Config()