from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import text
import os
from pathlib import Path
from dotenv import load_dotenv
import uuid
from pinecone_client import get_index
import json
from db import get_db
from auth import require_tenant, create_api_key
from ingest import split_text

load_dotenv()

app = FastAPI()


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Temporary bootstrap token so random people can’t create keys.
# Put it in .env as ADMIN_BOOTSTRAP_TOKEN=some-long-random-string
ADMIN_BOOTSTRAP_TOKEN = os.environ.get("ADMIN_BOOTSTRAP_TOKEN")


def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")

@app.post("/chat")
def chat(
    payload: dict,
    ctx=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    require_scope(ctx, "chat")
    org_id = str(ctx["org_id"])

    user_message = (payload.get("message") or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    index = get_index()

    # Pinecone hosted embedding search (text -> embedding internally)
    results = index.search(
        namespace=org_id,
        query={
            "inputs": {"text": user_message},
            "top_k": 5
        },
        fields=["document_id", "chunk_index", "filename"]
    )

    hits = results["result"]["hits"]
    match_ids = [h["_id"] for h in hits]

    rows = []

    # If nothing found in this tenant namespace
    if not match_ids:
        context = ""
    else:
        # Fetch chunk text from Postgres, still filtered by org_id
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

        # Keep Pinecone ranking order
        content_by_id = {r["id"]: r["content"] for r in rows}
        context = "\n\n---\n\n".join(content_by_id[i] for i in match_ids if i in content_by_id)


    # Get active avatar settings for this org (optional, but useful now)
    settings = db.execute(
        text("""
            SELECT tone, system_prompt, safety_prompt, avatar_id, voice_id
            FROM avatar_settings
            WHERE org_id = :org_id AND is_active = true
            LIMIT 1
        """),
        {"org_id": org_id},
    ).mappings().first()

    # For now we are not calling an LLM yet.
    # We return what would be sent to the LLM.
    return {
        "org_id": org_id,
        "retrieved_chunks": len(match_ids),
        "avatar_settings_found": bool(settings),
        "system_prompt": settings["system_prompt"] if settings else None,
        "tone": settings["tone"] if settings else None,
        "context_preview": context[:1200],  # just to keep response small
        "user_message": user_message,
        "next_step": "Integrate LLM + streaming; replace context_preview with real answer",
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    ctx=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    require_scope(ctx, "upload")

    org_id = str(ctx["org_id"])

    # For now: accept only text files (keeps it simple)
    if not file.content_type or not file.content_type.startswith("text/"):
        raise HTTPException(status_code=400, detail="Only text/* files supported for now")

    raw = await file.read()
    text_content = raw.decode("utf-8", errors="ignore").strip()
    if not text_content:
        raise HTTPException(status_code=400, detail="File has no readable text")

    # Save file locally (organized by org_id)
    org_folder = UPLOAD_DIR / org_id
    org_folder.mkdir(parents=True, exist_ok=True)

    doc_uuid = str(uuid.uuid4())
    safe_name = file.filename or "uploaded.txt"
    storage_path = org_folder / f"{doc_uuid}_{safe_name}"

    storage_path.write_text(text_content, encoding="utf-8")


    # Split into chunks and insert
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

        # 1) Insert into Postgres (store the SAME chunk_id)
        db.execute(
            text("""
                INSERT INTO chunks (id, org_id, document_id, chunk_index, content, metadata)
                VALUES (:id, :org_id, :document_id, :chunk_index, :content, CAST(:metadata AS jsonb))
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

        # 2) Prepare Pinecone record (text will be embedded by Pinecone)
        records.append({
            "id": chunk_id,
            "text": ch,                 # must match field_map text->text
            "document_id": doc_uuid,
            "chunk_index": i,
            "filename": safe_name,
        })

    db.commit()
    
    # Upsert all chunks in ONE call (faster)
    index.upsert_records(namespace=org_id, records=records)

    return {"document_id": doc_uuid, "chunks_inserted": len(chunks)}


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/me")
def me(ctx=Depends(require_tenant)):
    return ctx

@app.get("/orgs")
def list_orgs(db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT id, name, slug, status FROM organizations")).mappings().all()
    return {"count": len(rows), "items": rows}

@app.post("/keys/create")
def keys_create(
    org_id: str,
    name: str | None = None,
    admin_token: str | None = Header(None, alias="X-ADMIN-TOKEN"),
    db: Session = Depends(get_db),
):
    if not ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_BOOTSTRAP_TOKEN not set")

    if admin_token != ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # Ensure org exists
    exists = db.execute(
        text("SELECT 1 FROM organizations WHERE id = :id LIMIT 1"),
        {"id": org_id},
    ).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Organization not found")

    full_key = create_api_key(db, org_id=org_id, name=name)
    return {"api_key": full_key}  

