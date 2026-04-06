# 📄 AI Document Q&A API

Ask questions about your PDF or text documents in plain English.
Built with **FastAPI**, **LangChain**, **OpenAI**, and **ChromaDB** using the **RAG (Retrieval-Augmented Generation)** pattern.

## How It Works

```
User uploads PDF
      ↓
Document is split into overlapping chunks
      ↓
Each chunk is embedded (converted to a vector) via OpenAI
      ↓
Vectors stored in ChromaDB (local vector database)
      ↓
User asks a question
      ↓
Question is embedded → top 4 matching chunks retrieved from ChromaDB
      ↓
Chunks injected as context into GPT prompt
      ↓
GPT generates an answer grounded in the document
```

## Features

- 📤 Upload PDF, TXT, or Markdown files
- 🔍 Semantic search across all uploaded documents
- 🎯 Scope questions to a specific document via `doc_id`
- 📡 Streaming responses for chat-like UX
- 🗂️ List and delete documents
- 🧾 Source attribution — see which chunks were used to answer
- 🐳 Docker-ready, no external services needed (ChromaDB runs locally)

## Tech Stack

| Component     | Technology                    |
|---------------|-------------------------------|
| API           | FastAPI                       |
| AI Orchestration | LangChain (LCEL)           |
| LLM           | OpenAI GPT-4o-mini            |
| Embeddings    | OpenAI text-embedding-3-small |
| Vector Store  | ChromaDB (local)              |
| PDF Parsing   | PyPDF                         |

## Getting Started

### 1. Clone and install
```bash
pip install -r requirements.txt
```

### 2. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Get a key at: https://platform.openai.com/api-keys

### 3. Run the server
```bash
uvicorn app.main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive API explorer.

### Or run with Docker
```bash
docker-compose up --build
```

## API Usage

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf" \
  -F "description=Q3 Annual Report"
```

**Response:**
```json
{
  "doc_id": "a1b2c3d4e5f6",
  "filename": "report.pdf",
  "total_pages": 12,
  "total_chunks": 48,
  "status": "ingested"
}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the key findings?", "doc_id": "a1b2c3d4e5f6"}'
```

**Response:**
```json
{
  "question": "What were the key findings?",
  "answer": "The key findings were...",
  "sources": [
    {
      "filename": "report.pdf",
      "page": 3,
      "excerpt": "..."
    }
  ],
  "model": "gpt-4o-mini"
}
```

### Streaming response
```bash
curl -X POST http://localhost:8000/api/v1/qa/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise this document"}'
```

## API Endpoints

| Method | Endpoint                        | Description                    |
|--------|---------------------------------|--------------------------------|
| POST   | /api/v1/documents/upload        | Upload and index a document    |
| GET    | /api/v1/documents/              | List all indexed documents     |
| DELETE | /api/v1/documents/{doc_id}      | Delete a document              |
| POST   | /api/v1/qa/ask                  | Ask a question (full response) |
| POST   | /api/v1/qa/ask/stream           | Ask a question (streaming)     |
| GET    | /health                         | Health check                   |

## Running Tests
```bash
pytest tests/ -v
```

## Project Structure
```
app/
├── main.py
├── core/
│   ├── config.py        # Settings (env vars)
│   └── vectorstore.py   # ChromaDB initialisation
├── routers/
│   ├── documents.py     # Upload, list, delete
│   └── qa.py            # Ask, stream
└── services/
    ├── document_service.py  # Ingest pipeline
    └── qa_service.py        # RAG chain
```
