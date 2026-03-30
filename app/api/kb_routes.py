"""Knowledge-base API routes.

Endpoints
---------
POST  /api/v1/kb/upload        Upload a file and ingest it into the vector store
POST  /api/v1/kb/ingest_texts  Ingest raw text snippets directly
POST  /api/v1/kb/search        Semantic search (retrieve docs without LLM)
POST  /api/v1/kb/ask           Full RAG question-answering (retrieve + LLM generate)
DELETE /api/v1/kb/collection   Drop / clear the entire vector-store collection
GET   /api/v1/kb/health        Quick liveness check for the KB subsystem
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestTextsRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    metadatas: Optional[List[Dict[str, Any]]] = None


class IngestResponse(BaseModel):
    indexed: int
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=4, ge=1, le=20)


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=4, ge=1, le=20)
    session_id: str = Field(default="default")


class AskResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs_count: int
    sources: List[str]
    elapsed_seconds: float


class KBHealthResponse(BaseModel):
    status: str
    vector_store_type: str
    embedding_type: str
    embedding_model: str


class FileDetail(BaseModel):
    source: str
    chunk_count: int
    char_count: int


class KBInfoResponse(BaseModel):
    total_files: int
    total_chunks: int
    total_chars: int
    files: List[FileDetail]


# ---------------------------------------------------------------------------
# GET /kb/info  – knowledge-base document statistics
# ---------------------------------------------------------------------------

@router.get("/info", response_model=KBInfoResponse)
def kb_info():
    """Return per-file statistics of the current knowledge base.

    Response fields:
    - ``total_files``  – number of distinct source documents
    - ``total_chunks`` – total indexed chunk count
    - ``total_chars``  – total character count across all chunks
    - ``files``        – per-file breakdown list
    """
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        file_list = retriever.list_documents()
        files = [
            FileDetail(
                source=fi["source"],
                chunk_count=fi["chunk_count"],
                char_count=fi["char_count"],
            )
            for fi in file_list
        ]
        return KBInfoResponse(
            total_files=len(files),
            total_chunks=sum(f.chunk_count for f in files),
            total_chars=sum(f.char_count for f in files),
            files=files,
        )
    except Exception as exc:
        logger.error("[kb/info] %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve knowledge-base info: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /kb/health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=KBHealthResponse)
def kb_health():
    """Verify the knowledge-base subsystem is ready."""
    try:
        from app.rag.retriever import get_retriever
        get_retriever()  # triggers lazy init
        return KBHealthResponse(
            status="ok",
            vector_store_type=cfg.VECTOR_STORE_TYPE,
            embedding_type=cfg.EMBEDDING_TYPE,
            embedding_model=cfg.EMBEDDING_MODEL_NAME,
        )
    except Exception as exc:
        logger.error("[kb/health] %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Knowledge-base subsystem unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /kb/upload  – upload file and ingest
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(..., description="Document to ingest (.txt, .md, .pdf, .docx)"),
    chunk_size: int = Form(default=0, ge=0, description="Override chunk size (0 = use config)"),
    chunk_overlap: int = Form(default=0, ge=0, description="Override chunk overlap (0 = use config)"),
):
    """Upload a document file and store its chunks in the vector database.

    Supported formats: ``.txt``, ``.md``, ``.pdf``, ``.docx``

    The file is parsed, split into overlapping text chunks, embedded, and
    stored in the configured vector store (memory / Milvus / Chroma).
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    allowed_extensions = {".txt", ".md", ".pdf", ".docx"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    logger.info("[kb/upload] file=%s size=%d bytes", file.filename, len(raw))

    try:
        from app.rag.chunker import parse_and_chunk
        from app.rag.retriever import get_retriever

        chunks = parse_and_chunk(
            raw=raw,
            filename=file.filename,
            chunk_size=chunk_size or None,
            chunk_overlap=chunk_overlap or None,
        )
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File parsed but produced no text chunks. It may be empty or image-only.",
            )

        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]

        retriever = get_retriever()
        n = retriever.add_documents(texts, metas)

        return IngestResponse(
            indexed=n,
            message=f"Successfully indexed {n} chunks from '{file.filename}'.",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("[kb/upload] error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /kb/ingest_texts  – ingest raw text snippets
# ---------------------------------------------------------------------------

@router.post("/ingest_texts", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_texts(request: IngestTextsRequest):
    """Directly ingest raw text strings into the vector store (no file parsing)."""
    logger.info("[kb/ingest_texts] %d texts", len(request.texts))
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        n = retriever.add_documents(request.texts, request.metadatas)
        return IngestResponse(indexed=n, message=f"Successfully indexed {n} text chunks.")
    except Exception as exc:
        logger.error("[kb/ingest_texts] error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /kb/search  – semantic search only (no LLM)
# ---------------------------------------------------------------------------

@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """Retrieve the most relevant document chunks for a query without LLM generation."""
    logger.info("[kb/search] query=%r top_k=%d", request.query[:80], request.top_k)
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.retrieve(request.query, top_k=request.top_k)
        results = [
            SearchResult(
                content=d["content"],
                score=float(d["metadata"].get("score", 0.0)),
                metadata=d["metadata"],
            )
            for d in docs
        ]
        return SearchResponse(query=request.query, results=results, total=len(results))
    except Exception as exc:
        logger.error("[kb/search] error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /kb/ask  – full RAG: retrieve + LLM answer
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question using RAG: retrieve relevant chunks then generate an answer with the LLM.

    This endpoint bypasses the full LangGraph agent workflow and provides a
    direct retrieve-then-generate pipeline for knowledge-base Q&A.
    """
    import time
    start = time.perf_counter()
    logger.info("[kb/ask] session=%s query=%r", request.session_id, request.query[:80])

    try:
        from app.rag.retriever import get_retriever
        from app.llm.client import get_llm_client
        from app.prompts.templates import ANSWER_WITH_CONTEXT, ANSWER_NO_CONTEXT

        # 1. Retrieve
        retriever = get_retriever()
        docs = retriever.retrieve(request.query, top_k=request.top_k)
        logger.info("[kb/ask] retrieved %d docs", len(docs))

        # 2. Build prompt
        client = get_llm_client()
        if docs:
            context = "\n\n".join(
                f"[{i+1}] {d['content']}" for i, d in enumerate(docs)
            )
            prompt = ANSWER_WITH_CONTEXT.format(
                query=request.query,
                context=context,
                tool_result="N/A",
            )
        else:
            prompt = ANSWER_NO_CONTEXT.format(
                query=request.query,
                tool_result="N/A",
            )

        # 3. Generate
        answer = client.chat_sync([{"role": "user", "content": prompt}])
        elapsed = round(time.perf_counter() - start, 3)

        sources = list({
            d["metadata"].get("source", "") for d in docs
            if d["metadata"].get("source")
        })

        return AskResponse(
            query=request.query,
            answer=answer,
            retrieved_docs_count=len(docs),
            sources=sources,
            elapsed_seconds=elapsed,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[kb/ask] error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG QA failed: {exc}",
        )


# ---------------------------------------------------------------------------
# DELETE /kb/collection  – drop the entire vector store collection
# ---------------------------------------------------------------------------

@router.delete("/collection", status_code=status.HTTP_200_OK)
def delete_collection():
    """Drop and reset the entire vector-store collection.

    **Warning**: this permanently deletes all indexed documents.
    """
    logger.warning("[kb/collection] DELETE requested")
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        retriever.delete_collection()
        return {"message": f"Collection '{cfg.VECTOR_STORE_TYPE}' cleared successfully."}
    except Exception as exc:
        logger.error("[kb/collection] error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear collection: {exc}",
        )
