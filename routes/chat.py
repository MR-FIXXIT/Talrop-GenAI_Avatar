# routes/chat.py
from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from auth import require_tenant

from rag.chat_rag import chat_rag

router = APIRouter(tags=["chat"])


class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    chat_history: List[ChatMsg] = Field(default_factory=list)
    top_k: int = 5


def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")


@router.post("/chat")
def chat(payload: ChatRequest, ctx=Depends(require_tenant), db: Session = Depends(get_db)):
    require_scope(ctx, "chat")
    org_id = str(ctx["org_id"])

    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    # DB-only: settings lookup stays in route layer
    settings = db.execute(
        text("""
            SELECT tone, system_prompt, safety_prompt, avatar_id, voice_id
            FROM avatar_settings
            WHERE org_id = CAST(:org_id AS uuid) AND is_active = true
            LIMIT 1
        """),
        {"org_id": org_id},
    ).mappings().first()

    
    result = chat_rag(
        db,
        org_id=org_id,
        user_message=user_message,
        history=[m.model_dump() for m in payload.chat_history],
        org_system_prompt=(settings["system_prompt"] if settings else None),
        temperature=0.0,
        max_new_tokens=3000,
        top_k=payload.top_k,
    )

    return {
        "org_id": org_id,
        "question_used_for_retrieval": result.standalone_question,
        "retrieved_chunks": len(result.match_ids),
        "match_ids": result.match_ids,
        "avatar_settings_found": bool(settings),
        "tone": settings["tone"] if settings else None,
        "answer": result.answer,
        "context_preview": result.context[:],
    }
