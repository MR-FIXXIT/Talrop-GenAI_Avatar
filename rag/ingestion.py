# rag/ingestion.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import text

from pinecone_client import get_index

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# NEW: PDF loader (recommended if you already use LangChain)
from langchain_community.document_loaders import PyPDFLoader



@dataclass
class UploadResult:
    # A small structured return value:
    # - document_id: UUID for the uploaded document (stored in DB)
    # - chunks_inserted: number of semantic chunks created
    # - filename, storage_path, content_type: metadata about the stored file
    document_id: str
    chunks_inserted: int
    filename: str
    storage_path: str
    content_type: Optional[str]


def semantic_chunk_text(
    text: str,
    *,
    model_name: str,
    min_chunk_size: int,
    buffer_size: int = 1,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> list[str]:
    # Input: raw text string
    # Output: list of chunk strings
    # 
    # How it works:
    # - Normalizes line endings, strips whitespace
    # - Creates HuggingFaceEmbeddings(model_name=...) (here: sentence-transformers/all-MiniLM-L6-v2)
    # 
    # - Builds a SemanticChunker which:
    #   - embeds text
    #   - tries to split at “semantic breakpoints” (where meaning changes) rather than fixed-size windows
    #   - uses parameters like:
    #     - buffer_size=1: keeps a small context window when deciding splits
    #     - breakpoint_threshold_type="percentile" and amount=95.0: “only split at relatively strong semantic-change points”
    #     - min_chunk_size=200: avoids tiny chunks
    # 
    # - Returns the cleaned chunk contents
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    splitter = SemanticChunker(
        embeddings=embeddings,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
    )

    docs = splitter.create_documents([text])
    return [d.page_content.strip() for d in docs if d.page_content and d.page_content.strip()]


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF on disk.
    Note: scanned/image-only PDFs will often return empty text without OCR.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        text_content = "\n\n".join(d.page_content for d in docs if d.page_content)
        return (text_content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {e}")


def ingest_and_index_text_file(
    *,
    db: Session,
    org_id: str,
    file: UploadFile,
    upload_dir: Path,
) -> UploadResult:
    # ---- validate type (NOW supports text/* and PDFs) ----
    filename = (file.filename or "uploaded").strip()
    content_type = file.content_type or ""

    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_text = content_type.startswith("text/")

    if not (is_text or is_pdf):
        raise HTTPException(
            status_code=400,
            detail="Only text/* and application/pdf files supported for now",
        )

    # ---- read raw bytes once ----
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    # ---- save to disk ----
    org_folder = upload_dir / org_id
    org_folder.mkdir(parents=True, exist_ok=True)

    doc_uuid = str(uuid.uuid4())
    safe_name = filename or ("uploaded.pdf" if is_pdf else "uploaded.txt")
    storage_path = org_folder / f"{doc_uuid}_{safe_name}"

    if is_pdf:
        storage_path.write_bytes(raw)
        text_content = _extract_text_from_pdf(storage_path)
    else:
        text_content = raw.decode("utf-8", errors="ignore").strip()
        storage_path.write_text(text_content, encoding="utf-8")

    if not text_content:
        # Common for scanned PDFs (image-only) without OCR
        raise HTTPException(status_code=400, detail="File has no readable text")

    # ---- create documents row (required for chunks FK) ----
    db.execute(
        text("""
            INSERT INTO documents (id, org_id, filename, content_type, storage_path, status)
            VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                :filename,
                :content_type,
                :storage_path,
                'ready'
            )
        """),
        {
            "id": doc_uuid,
            "org_id": org_id,
            "filename": safe_name,
            "content_type": (content_type or ("application/pdf" if is_pdf else "text/plain")),
            "storage_path": str(storage_path),
        },
    )

    # ---- semantic chunk ----
    chunks = semantic_chunk_text(
        text_content,
        min_chunk_size=200,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # ---- insert chunks + build pinecone records with SAME chunk_id ----
    index = get_index()
    records: List[Dict[str, Any]] = []

    for i, ch in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        meta = {
            "org_id": org_id,
            "document_id": doc_uuid,
            "chunk_index": i,
            "filename": safe_name,
        }

        db.execute(
            text("""
                INSERT INTO chunks (id, org_id, document_id, chunk_index, content, metadata)
                VALUES (
                    CAST(:id AS uuid),
                    CAST(:org_id AS uuid),
                    CAST(:document_id AS uuid),
                    :chunk_index,
                    :content,
                    CAST(:metadata AS jsonb)
                )
            """),
            {
                "id": chunk_id,
                "org_id": org_id,
                "document_id": doc_uuid,
                "chunk_index": i,
                "content": ch,
                "metadata": json.dumps(meta),
            },
        )

        records.append(
            {
                "id": chunk_id,
                "text": ch,
                "document_id": doc_uuid,
                "chunk_index": i,
                "filename": safe_name,
            }
        )

    db.commit()
    index.upsert_records(namespace=org_id, records=records)

    return UploadResult(
        document_id=doc_uuid,
        chunks_inserted=len(chunks),
        filename=safe_name,
        storage_path=str(storage_path),
        content_type=(content_type or ("application/pdf" if is_pdf else "text/plain")),
    )
