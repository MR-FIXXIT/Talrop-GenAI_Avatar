import uuid
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from auth import require_tenant
from ingest import split_text
from pinecone_client import get_index

router = APIRouter(tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    ctx=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    require_scope(ctx, "upload")
    org_id = str(ctx["org_id"])

    if not file.content_type or not file.content_type.startswith("text/"):
        raise HTTPException(status_code=400, detail="Only text/* files supported for now")

    raw = await file.read()
    text_content = raw.decode("utf-8", errors="ignore").strip()
    if not text_content:
        raise HTTPException(status_code=400, detail="File has no readable text")

    org_folder = UPLOAD_DIR / org_id
    org_folder.mkdir(parents=True, exist_ok=True)

    doc_uuid = str(uuid.uuid4())
    safe_name = file.filename or "uploaded.txt"
    storage_path = org_folder / f"{doc_uuid}_{safe_name}"
    storage_path.write_text(text_content, encoding="utf-8")


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
            "content_type": file.content_type,
            "storage_path": str(storage_path),
        },
    )


    chunks = split_text(text_content, chunk_size=800, overlap=120)

    index = get_index()
    records = []

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
                VALUES (:id, CAST(:org_id AS uuid), CAST(:document_id AS uuid), :chunk_index, :content, CAST(:metadata AS jsonb))
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

        records.append({
            "id": chunk_id,
            "text": ch,
            "document_id": doc_uuid,
            "chunk_index": i,
            "filename": safe_name,
        })

    db.commit()
    index.upsert_records(namespace=org_id, records=records)

    return {"document_id": doc_uuid, "chunks_inserted": len(chunks)}
