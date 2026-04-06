from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.services.qa_service import QAService

router = APIRouter()
svc = QAService()


class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None   # if None, searches across ALL documents

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the main findings of this report?",
                    "doc_id": None,
                },
                {
                    "question": "What is the conclusion?",
                    "doc_id": "abc123def456",
                },
            ]
        }
    }


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list
    doc_id: Optional[str]
    model: str


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(payload: QuestionRequest):
    """
    Ask a question about your uploaded documents.

    - If **doc_id** is provided, only that document is searched.
    - If **doc_id** is omitted, all uploaded documents are searched.

    Returns the answer plus source excerpts showing which parts of the
    document were used to generate the response.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = await svc.answer(
            question=payload.question,
            doc_id=payload.doc_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")

    return result


@router.post("/ask/stream")
async def ask_question_stream(payload: QuestionRequest):
    """
    Streaming version of /ask.
    Returns answer tokens as they are generated — ideal for chat-like UIs.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def token_generator():
        try:
            async for token in svc.answer_stream(
                question=payload.question,
                doc_id=payload.doc_id,
            ):
                yield token
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
