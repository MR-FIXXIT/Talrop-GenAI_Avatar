# rag/chat_rag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

from sqlalchemy.orm import Session

from rag.retriever import retrieve_context
from rag.generator import (
    HistoryItem,
    build_llm,
    history_to_lc,
    contextualize_question,
    build_effective_system_prompt,
    generate_answer,
    strip_think,
)


Role = Literal["user", "assistant"]


@dataclass
class RagResult:
    standalone_question: str
    match_ids: List[str]
    context: str
    answer: str


def chat_rag(
    db: Session,
    *,
    org_id: str,
    user_message: str,
    history: Sequence[HistoryItem],
    org_system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    top_k: int = 5,
    min_score: float = 0.3,
    strip_model_think_tags: bool = True,
) -> RagResult:
    """
    Orchestrates the RAG flow:
      1) build llm
      2) history -> lc format
      3) contextualize question
      4) retrieve context
      5) build system prompt
      6) generate answer
      7) return RagResult
    """
    llm = build_llm(temperature=temperature, max_new_tokens=max_new_tokens)
    lc_history = history_to_lc(history)

    standalone_question = contextualize_question(
        llm,
        user_message=user_message,
        lc_history=lc_history,
    )

    context, match_ids = retrieve_context(
        db,
        org_id=org_id,
        question=standalone_question,
        top_k=top_k,
        min_score=min_score,
    )

    system_prompt = build_effective_system_prompt(org_system_prompt=org_system_prompt)

    answer = generate_answer(
        llm,
        system_prompt=system_prompt,
        user_message=user_message,
        lc_history=lc_history,
        context=context,
    )

    if strip_model_think_tags:
        answer = strip_think(answer)

    # Enforce the contract if model returns empty
    answer = (answer or "").strip() or "I don't know."

    return RagResult(
        standalone_question=standalone_question,
        match_ids=match_ids,
        context=context,
        answer=answer,
    )