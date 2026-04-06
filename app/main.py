from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import documents, qa
from app.core.config import settings
from app.core.vectorstore import init_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize ChromaDB vector store on startup
    init_vectorstore()
    yield


app = FastAPI(
    title="AI Document Q&A API",
    description="""
    Upload PDF or text documents and ask questions about them in natural language.
    Powered by LangChain, OpenAI, and ChromaDB using RAG (Retrieval-Augmented Generation).
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(qa.router,        prefix="/api/v1/qa",        tags=["Q&A"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Document Q&A API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
