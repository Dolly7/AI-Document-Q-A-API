import os
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings
from app.core.vectorstore import get_vectorstore


class DocumentService:
    """
    Handles the full document ingestion pipeline:
    1. Save uploaded file to disk
    2. Load and parse document (PDF or text)
    3. Split into chunks
    4. Embed and store in ChromaDB
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def validate_file(self, filename: str, file_size: int) -> None:
        ext = Path(filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '{ext}' not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB")

    def save_file(self, filename: str, content: bytes) -> str:
        """Save uploaded file, return its saved path."""
        safe_name = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(settings.UPLOAD_DIR, safe_name)
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def load_documents(self, filepath: str) -> list[Document]:
        """Load and parse a PDF or text file into LangChain Documents."""
        ext = Path(filepath).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath, encoding="utf-8")
        return loader.load()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into overlapping chunks for better retrieval."""
        return self.text_splitter.split_documents(documents)

    def compute_doc_id(self, content: bytes) -> str:
        """Generate a stable document ID from file content hash."""
        return hashlib.md5(content).hexdigest()

    async def ingest(
        self, filename: str, content: bytes, description: Optional[str] = None
    ) -> dict:
        """
        Full ingestion pipeline:
        save → load → split → embed → store in ChromaDB
        """
        # 1. Save to disk
        filepath = self.save_file(filename, content)
        doc_id = self.compute_doc_id(content)

        # 2. Load
        raw_docs = self.load_documents(filepath)

        # 3. Split into chunks
        chunks = self.split_documents(raw_docs)

        # 4. Attach metadata to every chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id":      doc_id,
                "filename":    filename,
                "chunk_index": i,
                "description": description or "",
            })

        # 5. Store in ChromaDB (auto-embeds via OpenAI)
        vectorstore = get_vectorstore()
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=chunk_ids)

        return {
            "doc_id":      doc_id,
            "filename":    filename,
            "total_pages": len(raw_docs),
            "total_chunks": len(chunks),
            "description": description or "",
            "status":      "ingested",
        }

    def list_documents(self) -> list[dict]:
        """List all unique documents stored in ChromaDB."""
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])

        seen = {}
        for meta in results.get("metadatas", []):
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id":      doc_id,
                    "filename":    meta.get("filename", "unknown"),
                    "description": meta.get("description", ""),
                }
        return list(seen.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to a document from ChromaDB."""
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        results = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = results.get("ids", [])
        if not ids:
            return False
        collection.delete(ids=ids)
        return True
