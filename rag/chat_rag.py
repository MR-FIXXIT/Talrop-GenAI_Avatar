#rag/chat_rag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from rag.retriever import retrieve_context, RetrievedChunk

from rag.generator import (
    HistoryItem,
    build_llm,
    history_to_lc,
    contextualize_question,
    extract_supported_facts,
    generate_answer_from_facts,
    revise_answer_for_faithfulness,
    normalize_final_answer,
)

import rag.prompts as prompts


Role = Literal["user", "assistant"]

_NO_CONTEXT_ANSWER = "Answer: I don't know based on the provided context."


@dataclass
class RagResult:
    standalone_question: str
    context: str
    answer: str
    supported_facts: str


def format_labeled_context(chunks: List[RetrievedChunk]) -> str:
    parts: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        text = (ch.text or "").strip()
        if not text:
            continue
        parts.append(f"[c{i}] (chunk_source={ch.file_name}, score={ch.score:.4f})\n{text}")
    return "\n\n".join(parts).strip()


def chat_rag(
    *,
    org_id: str,
    user_message: str,
    history: Sequence[HistoryItem],
    org_system_prompt: Optional[str] = None,
    temperature: float,
    max_new_tokens: int,
    top_k: int = 5,
    min_score: float = 0.1,
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

    # Recommended retriever contract:
    # returns List[RetrievedChunk]
    chunks = retrieve_context(
        org_id=org_id,
        question=standalone_question,
        top_k=top_k,
        min_score=min_score,
    )

    if not chunks:
        return RagResult(
            standalone_question=standalone_question,
            context="",
            answer=_NO_CONTEXT_ANSWER,
            supported_facts="NO_SUPPORT",
        )

    # match_ids = [c.chunk_id for c in chunks]
    labeled_context = format_labeled_context(chunks)

    fact_extract_system_prompt = prompts.build_effective_fact_extract_system_prompt(
        org_system_prompt=org_system_prompt,
        context=labeled_context,
    )

    supported_facts = extract_supported_facts(
        llm,
        system_prompt=fact_extract_system_prompt,
        question=standalone_question,
    )

    print(f"Supported Facts : {supported_facts}")
    
    if not supported_facts or supported_facts.strip() == "NO_SUPPORT":
        answer = _NO_CONTEXT_ANSWER
    else:
        answer_from_facts_system_prompt = (
            prompts.build_effective_answer_from_facts_system_prompt(
                org_system_prompt=org_system_prompt,
                context=labeled_context,
                supported_facts=supported_facts,
            )
        )


        draft_answer = generate_answer_from_facts(
            llm,
            system_prompt=answer_from_facts_system_prompt,
            question=standalone_question,
        )

        print(f"Draft Answer : {draft_answer}")

        revision_system_prompt = (
            prompts.build_effective_faithfulness_revision_system_prompt(
                org_system_prompt=org_system_prompt,
                context=labeled_context,
                draft_answer=draft_answer,
            )
        )

        answer = revise_answer_for_faithfulness(
            llm,
            system_prompt=revision_system_prompt,
            question=standalone_question,
        )

        print(f"Answer : {answer}")

    answer = normalize_final_answer(answer)

    return RagResult(
        standalone_question=standalone_question,
        context=labeled_context,
        answer=answer,
        supported_facts=supported_facts,
    )