import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

# Module-level singletons
_chroma_client = None
_vectorstore = None
_embeddings = None


def init_vectorstore():
    """Initialize ChromaDB and embeddings on app startup."""
    global _chroma_client, _vectorstore, _embeddings

    _embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    _vectorstore = Chroma(
        client=_chroma_client,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=_embeddings,
    )

    print(f"✅ ChromaDB initialized at {settings.CHROMA_PERSIST_DIR}")


def get_vectorstore() -> Chroma:
    if _vectorstore is None:
        init_vectorstore()
    return _vectorstore


def get_embeddings() -> OpenAIEmbeddings:
    if _embeddings is None:
        init_vectorstore()
    return _embeddings
