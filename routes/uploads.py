from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from auth import require_tenant

from rag.ingestion import ingest_and_index



router = APIRouter(tags=["uploads"])


def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    ctx=Depends(require_tenant)
):
    require_scope(ctx, "upload")
    org_id = str(ctx["org_id"])

    result = ingest_and_index(
        file=file,
        org_id=org_id
    )

    return {
        "document_id": result.document_id,
        # "chunks_inserted": result.chunks_inserted,
    }
