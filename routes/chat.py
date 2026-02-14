from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from auth import require_tenant
from pinecone_client import get_index

router = APIRouter(tags=["chat"])

def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")

@router.post("/chat")
def chat(payload: dict, ctx=Depends(require_tenant), db: Session = Depends(get_db)):
    require_scope(ctx, "chat")
    org_id = str(ctx["org_id"])

    user_message = (payload.get("message") or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    index = get_index()
    results = index.search(
        namespace=org_id,
        query={"inputs": {"text": user_message}, "top_k": 5},
        fields=["document_id", "chunk_index", "filename"],
    )

    hits = results["result"]["hits"]
    match_ids = [h["_id"] for h in hits]

    if not match_ids:
        context = ""
    else:
        placeholders = ", ".join([f":id{i}" for i in range(len(match_ids))])
        params = {"org_id": org_id, **{f"id{i}": match_ids[i] for i in range(len(match_ids))}}

        rows = db.execute(
            text(f"""
                SELECT id, content
                FROM chunks
                WHERE org_id = CAST(:org_id AS uuid)
                AND id IN ({placeholders})
            """),
            params,
        ).mappings().all()

        # IMPORTANT: normalize DB UUID -> string so it matches Pinecone ids
        content_by_id = {str(r["id"]): r["content"] for r in rows}
        context = "\n\n---\n\n".join(content_by_id[i] for i in match_ids if i in content_by_id)

    settings = db.execute(
        text("""
            SELECT tone, system_prompt, safety_prompt, avatar_id, voice_id
            FROM avatar_settings
            WHERE org_id = CAST(:org_id AS uuid) AND is_active = true
            LIMIT 1
        """),
        {"org_id": org_id},
    ).mappings().first()

    return {
        "org_id": org_id,
        "retrieved_chunks": len(match_ids),
        "avatar_settings_found": bool(settings),
        "system_prompt": settings["system_prompt"] if settings else None,
        "tone": settings["tone"] if settings else None,
        "context_preview": context[:1200],
        "user_message": user_message,
    }
