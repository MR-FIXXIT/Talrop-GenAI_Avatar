# rag/ingestion.py
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from pinecone_client import get_index
from rag.textbook_chunker import preprocess_pdf, preprocess_text

# Embedding model
from sentence_transformers import SentenceTransformer

# Load once (global)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class UploadResult:
    document_id: str
    filename: str
    content_type: Optional[str]


def ingest_and_index(
    *,
    org_id: str,
    file: UploadFile,
    num_sentence_chunk_size: int = 10,
    min_token_length: int = 30
) -> UploadResult:

    filename = file.filename.strip()
    content_type = file.content_type or ""

    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_text = content_type.startswith("text/") or filename.lower().endswith(".txt")
    
    if not (is_pdf or is_text):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and text files are supported",
        )

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    doc_uuid = str(uuid.uuid4())

    # ---- preprocess ----
    try:
        if is_pdf:
            chunks_df = preprocess_pdf(
                pdf_bytes=raw,
                num_sentence_chunk_size=num_sentence_chunk_size,
                min_token_length=min_token_length,
            )
        else:
            # Assume text
            text_str = raw.decode("utf-8", errors="ignore")
            chunks_df = preprocess_text(
                text=text_str,
                num_sentence_chunk_size=num_sentence_chunk_size,
                min_token_length=min_token_length,
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not preprocess file: {e}",
        )

    if chunks_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid chunks were produced from the file",
        )

    index = get_index()

    texts = []
    metadatas = []
    ids = []

    # ---- prepare data ----
    for i, row in chunks_df.reset_index(drop=True).iterrows():
        chunk_text = str(row["sentence_chunk"]).strip()
        if not chunk_text:
            continue

        chunk_id = str(uuid.uuid4())

        texts.append(chunk_text)
        ids.append(chunk_id)
        metadatas.append({
            "text": chunk_text,
            "document_id": doc_uuid,
            "chunk_index": i,
            "filename": filename,
            "page_number": int(row["page_number"]),
        })

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="No valid chunks after filtering",
        )

    # ---- generate embeddings ----
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # ---- upsert to Pinecone ----
    vectors = []
    for i in range(len(texts)):
        vectors.append({
            "id": ids[i],
            "values": embeddings[i].tolist(),
            "metadata": metadatas[i],
        })

    for batch in batch_list(vectors, 20):
        index.upsert(namespace=org_id, vectors=batch)

    return UploadResult(
        document_id=doc_uuid,
        filename=filename,
        content_type=content_type or ("application/pdf" if is_pdf else "text/plain"),
    )


def batch_list(items: list, batch_size: int) -> list[list]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]