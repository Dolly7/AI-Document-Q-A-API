from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Document Q&A API"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"        # cheap & capable for beginners
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ChromaDB (local vector store — no external service needed)
    CHROMA_PERSIST_DIR: str = "./data/vectorstore"
    CHROMA_COLLECTION_NAME: str = "documents"

    # File upload
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".md"]

    # RAG settings
    CHUNK_SIZE: int = 1000       # characters per chunk
    CHUNK_OVERLAP: int = 200     # overlap between chunks
    TOP_K_RESULTS: int = 4       # number of chunks to retrieve per query

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
