# routes/documents.py
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from db import get_db
from auth import require_tenant
from rag.ingestion import ingest_and_index_text_file

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

    result = ingest_and_index_text_file(
        db=db,
        org_id=org_id,
        file=file,
        upload_dir=UPLOAD_DIR
    )

    return {
        "document_id": result.document_id,
        "chunks_inserted": result.chunks_inserted,
    }
