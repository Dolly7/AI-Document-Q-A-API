from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.core.vectorstore import get_vectorstore


# ── Prompt template ────────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based on the provided document context.

Use ONLY the information from the context below to answer the question.
If the answer is not in the context, say "I couldn't find that information in the uploaded document."
Be concise and accurate. If helpful, mention which part of the document your answer comes from.

Context:
{context}

Question: {question}

Answer:""",
)


class QAService:
    """
    RAG (Retrieval-Augmented Generation) Q&A service.

    Flow:
    1. Convert user question to an embedding vector
    2. Retrieve the top-K most relevant chunks from ChromaDB
    3. Inject retrieved chunks into the prompt as context
    4. Send prompt to OpenAI GPT → return answer
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,               # deterministic answers
            openai_api_key=settings.OPENAI_API_KEY,
        )

    def _build_retriever(self, doc_id: Optional[str] = None):
        """Build a retriever, optionally filtered to a specific document."""
        vectorstore = get_vectorstore()
        search_kwargs = {"k": settings.TOP_K_RESULTS}
        if doc_id:
            search_kwargs["filter"] = {"doc_id": doc_id}
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    def _format_docs(self, docs) -> str:
        """Format retrieved chunks into a single context string."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "document")
            page   = doc.metadata.get("page", "?")
            parts.append(f"[Chunk {i} — {source}, page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    async def answer(
        self,
        question: str,
        doc_id: Optional[str] = None,
    ) -> dict:
        """
        Answer a question using RAG.
        Optionally scope retrieval to a specific document via doc_id.
        """
        retriever = self._build_retriever(doc_id)

        # Build LCEL (LangChain Expression Language) chain
        chain = (
            {
                "context":  retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        answer_text = await chain.ainvoke(question)

        # Also retrieve source chunks for transparency
        source_docs = retriever.invoke(question)
        sources = [
            {
                "filename":    doc.metadata.get("filename"),
                "page":        doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "excerpt":     doc.page_content[:200] + "...",
            }
            for doc in source_docs
        ]

        return {
            "question": question,
            "answer":   answer_text,
            "sources":  sources,
            "doc_id":   doc_id,
            "model":    settings.OPENAI_MODEL,
        }

    async def answer_stream(self, question: str, doc_id: Optional[str] = None):
        """
        Streaming version — yields answer tokens as they arrive.
        Use with FastAPI's StreamingResponse.
        """
        retriever = self._build_retriever(doc_id)
        source_docs = retriever.invoke(question)
        context = self._format_docs(source_docs)

        stream_llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True,
        )

        prompt_value = QA_PROMPT.format(context=context, question=question)
        async for chunk in stream_llm.astream(prompt_value):
            yield chunk.content
