# routes/chat.py
from typing import List, Literal
import re
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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
    evaluate: bool = False
    top_k: int = 3


def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")



def run_evaluation(query: str, generated_answer: str, retrieved_context: str):
    try:
        from eval.evaluation import(
            evaluate_generation, 
            evaluate_retrieval, 
            evaluate_pipeline
        ) 

        #split each retrieved chunk into individual string in a list
        pattern = r'\[c\d+\]\s\(source_id=[^)]*\)'
        parts = re.split(pattern, retrieved_context)
        chunks = [part.strip() for part in parts if part.strip()]


        # generation_eval_result = evaluate_generation(
        #     query=query,
        #     generated_answer=generated_answer,
        #     context=chunks
        # )

        # retrieval_eval_result = evaluate_retrieval(
        #     query=query,
        #     generated_answer=generated_answer,
        #     context=chunks
        # )

        eval_result = evaluate_pipeline(
            query=query,
            generated_answer=generated_answer,
            context=chunks
        )
        

        path = f'C:/Users/banuv/Desktop/Talrop-GenAI_Avatar/eval/result_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.json'

        with open(path, "w") as f:
            json.dump(eval_result, f, indent=2)

    except Exception as e:
        print("Evaluation failed")
        print(str(e))



@router.post("/chat")
def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    ctx=Depends(require_tenant),
    db: Session = Depends(get_db)
):
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
        org_id=org_id,
        user_message=user_message,
        history=[m.model_dump() for m in payload.chat_history],
        org_system_prompt=(settings["system_prompt"] if settings else None),
        temperature=0.0,
        max_new_tokens=3000,
        top_k=payload.top_k,
    )

    #run evaluation after returning the endpoint
    if payload.evaluate:
        background_tasks.add_task(
            run_evaluation,
            query=payload.message,
            generated_answer=result.answer,
            retrieved_context=result.context,
        )


    return {
        "org_id": org_id,
        "question_used_for_retrieval": result.standalone_question,
        "avatar_settings_found": bool(settings),
        "tone": settings["tone"] if settings else None,
        "answer": result.answer,
        "context_preview": result.context[:],
    }
