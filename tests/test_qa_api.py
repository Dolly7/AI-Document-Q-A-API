import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    with patch("app.routers.documents.svc") as mock_svc:
        mock_svc.list_documents.return_value = []
        r = await client.get("/api/v1/documents/")
    assert r.status_code == 200
    assert r.json()["total"] == 0


@pytest.mark.asyncio
async def test_upload_invalid_extension(client):
    with patch("app.routers.documents.svc") as mock_svc:
        mock_svc.validate_file.side_effect = ValueError("File type '.exe' not supported.")
        r = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("malware.exe", b"bad content", "application/octet-stream")},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_pdf(client):
    with patch("app.routers.documents.svc") as mock_svc:
        mock_svc.validate_file.return_value = None
        mock_svc.ingest = AsyncMock(return_value={
            "doc_id": "abc123",
            "filename": "report.pdf",
            "total_pages": 5,
            "total_chunks": 20,
            "description": "Test report",
            "status": "ingested",
        })
        r = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.pdf", b"%PDF fake content", "application/pdf")},
            data={"description": "Test report"},
        )
    assert r.status_code == 201
    assert r.json()["doc_id"] == "abc123"
    assert r.json()["total_chunks"] == 20


@pytest.mark.asyncio
async def test_ask_question(client):
    with patch("app.routers.qa.svc") as mock_svc:
        mock_svc.answer = AsyncMock(return_value={
            "question": "What is this document about?",
            "answer": "This document is about machine learning.",
            "sources": [{"filename": "report.pdf", "page": 1, "chunk_index": 0, "excerpt": "..."}],
            "doc_id": None,
            "model": "gpt-4o-mini",
        })
        r = await client.post("/api/v1/qa/ask", json={
            "question": "What is this document about?"
        })
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["sources"]) > 0


@pytest.mark.asyncio
async def test_ask_empty_question(client):
    r = await client.post("/api/v1/qa/ask", json={"question": "   "})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_delete_document_not_found(client):
    with patch("app.routers.documents.svc") as mock_svc:
        mock_svc.delete_document.return_value = False
        r = await client.delete("/api/v1/documents/nonexistent-id")
    assert r.status_code == 404


# Unit tests — no HTTP needed
def test_document_service_validate_allowed():
    from app.services.document_service import DocumentService
    svc = DocumentService()
    svc.validate_file("report.pdf", 1024 * 1024)  # 1MB — should not raise


def test_document_service_validate_bad_extension():
    from app.services.document_service import DocumentService
    svc = DocumentService()
    with pytest.raises(ValueError, match="not supported"):
        svc.validate_file("virus.exe", 100)


def test_document_service_validate_too_large():
    from app.services.document_service import DocumentService
    svc = DocumentService()
    with pytest.raises(ValueError, match="too large"):
        svc.validate_file("big.pdf", 999 * 1024 * 1024)


def test_compute_doc_id_deterministic():
    from app.services.document_service import DocumentService
    svc = DocumentService()
    content = b"hello world"
    assert svc.compute_doc_id(content) == svc.compute_doc_id(content)
