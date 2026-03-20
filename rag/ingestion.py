# rag/ingestion.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from pinecone_client import get_index

from rag.textbook_chunker import preprocess_textbook_pdf


@dataclass
class UploadResult:
    document_id: str
    # chunks_inserted: int
    filename: str
    content_type: Optional[str]


def ingest_and_index(
    *,
    org_id: str,
    file: UploadFile,
    # upload_dir: Path,
    num_sentence_chunk_size: int = 10,
    min_token_length: int = 30,
) -> UploadResult:

    # ---- validate type ----
    filename = file.filename.strip()
    content_type = file.content_type or ""

    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    if not is_pdf:
        raise HTTPException(
            status_code=400,
            detail="This ingestion path supports only PDF files",
        )

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    
    doc_uuid = str(uuid.uuid4())
    
    # ---- run your textbook preprocessing pipeline ----
    try:
        chunks_df = preprocess_textbook_pdf(
            pdf_bytes=raw,
            num_sentence_chunk_size=num_sentence_chunk_size,
            min_token_length=min_token_length,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not preprocess textbook PDF: {e}",
        )

    if chunks_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid chunks were produced from the PDF",
        )

    # ---- insert chunks + build pinecone records with SAME chunk_id ----
    index = get_index()
    records: List[Dict[str, Any]] = []

    for i, row in chunks_df.reset_index(drop=True).iterrows():
        chunk_text = str(row["sentence_chunk"]).strip()
        if not chunk_text:
            continue

        chunk_id = str(uuid.uuid4())

        records.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "document_id": doc_uuid,
                "chunk_index": i,
                "filename": filename,
                "page_number": row["page_number"],
            }
        )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="Preprocessing ran, but no non-empty chunks were available for insertion",
        )

    for batch in batch_list(records, 10):
        index.upsert_records(namespace=org_id, records=batch)

    return UploadResult(
        document_id=doc_uuid,
        # chunks_inserted=len(records),
        filename=filename,
        content_type="application/pdf",
    )


def batch_list(items: list, batch_size: int) -> list[list]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]





# if __name__ == "__main__":
#     pdf_path = Path("C:/Users/banuv/Desktop/wegnio/CUSAT/SEM 2/dip/Dip textbook 4th edition.pdf")

#     with open(pdf_path, "rb") as f:
#         upload_file = UploadFile(
#             filename=pdf_path.name,
#             file=f,
#             headers={"content-type": "application/pdf"},
#         )

#         result = ingest_and_index(
#             org_id="02ba5304-1dfc-43d2-9283-aff179138d83",
#             file=upload_file,
#         )

#     print(result)