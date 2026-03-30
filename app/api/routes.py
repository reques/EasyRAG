"""FastAPI route definitions."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.agent_service import get_agent_service

logger = get_logger(__name__)
cfg = get_settings()
router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    query: str
    session_id: str
    intent: str
    intent_confidence: float
    retrieval_triggered: bool
    retrieved_docs_count: int
    tool_triggered: bool
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None
    tool_error: Optional[str] = None
    sub_tasks: List[str]
    steps: List[str]
    validation_passed: bool
    validation_feedback: str
    is_fallback: bool
    final_answer: str
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_type: str
    embedding_type: str
    llm_model: str


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class IngestResponse(BaseModel):
    indexed: int
    message: str


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    return HealthResponse(
        status="ok",
        version=cfg.APP_VERSION,
        vector_store_type=cfg.VECTOR_STORE_TYPE,
        embedding_type=cfg.EMBEDDING_TYPE,
        llm_model=cfg.LLM_MODEL,
    )


@router.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(request: ChatRequest):
    """Run the full agent workflow.

    Example request body::

        {"query": "What is Milvus?", "session_id": "demo_001"}
    """
    logger.info("/chat session=%s query=%r", request.session_id, request.query[:80])
    try:
        result = get_agent_service().run(
            query=request.query, session_id=request.session_id
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.error("/chat error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


@router.post("/ingest", response_model=IngestResponse, tags=["rag"])
def ingest(request: IngestRequest):
    """Index text chunks into the vector store."""
    logger.info("/ingest %d chunks", len(request.texts))
    try:
        from app.rag.retriever import get_retriever
        n = get_retriever().add_documents(request.texts, request.metadatas)
        return IngestResponse(indexed=n,
                              message="Successfully indexed " + str(n) + " chunks.")
    except Exception as exc:
        logger.error("/ingest error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


@router.get("/tools", tags=["agent"])
def list_tools():
    """List registered tools and their schemas."""
    from app.tools.registry import get_tool_registry
    reg = get_tool_registry()
    return {"tools": reg.list_names(), "schemas": reg.to_llm_schema()}
