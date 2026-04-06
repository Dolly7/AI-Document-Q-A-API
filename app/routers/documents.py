from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional

from app.services.document_service import DocumentService

router = APIRouter()
svc = DocumentService()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="PDF or text file to upload"),
    description: Optional[str] = Form(None, description="Optional description of the document"),
):
    """
    Upload a document (PDF, TXT, or MD).
    The document is automatically chunked, embedded, and stored
    in ChromaDB for semantic search.
    """
    content = await file.read()

    try:
        svc.validate_file(file.filename, len(content))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await svc.ingest(
            filename=file.filename,
            content=content,
            description=description,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {
        "message": "Document uploaded and indexed successfully",
        **result,
    }


@router.get("/")
async def list_documents():
    """List all documents currently stored in the vector store."""
    docs = svc.list_documents()
    return {
        "total": len(docs),
        "documents": docs,
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the vector store.
    Use the doc_id returned during upload.
    """
    deleted = svc.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully", "doc_id": doc_id}
