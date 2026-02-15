# rag/chat_rag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import re

from sqlalchemy.orm import Session
from sqlalchemy import text

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from pinecone_client import get_index


QA_SYSTEM_PROMPT = """You are an educational assistant and patient teacher.
Explain concepts thoroughly and break down complex ideas into simpler parts.
Use examples and analogies when helpful.
Use only the following pieces of retrieved context to answer the question.
If the answer is not contained within the retrieved context, say you don't know.

Context:
{context}"""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""


Role = Literal["user", "assistant"]
HistoryItem = Dict[str, str]  # {"role": "...", "content": "..."}


def build_llm(*, temperature: float = 0.7, max_new_tokens: int = 512) -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceTB/SmolLM3-3B",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=endpoint)


def history_to_lc(history: Sequence[HistoryItem]) -> List[Tuple[str, str]]:
    """
    Convert [{"role": "user"/"assistant", "content": "..."}]
    to LangChain chat messages format [("human", "..."), ("ai", "...")].
    """
    out: List[Tuple[str, str]] = []
    for m in history:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append(("human", content))
        else:
            out.append(("ai", content))
    return out


def contextualize_question(
    llm: ChatHuggingFace,
    *,
    user_message: str,
    lc_history: List[Tuple[str, str]],
) -> str:
    if not lc_history:
        return user_message

    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    q = chain.invoke({"chat_history": lc_history, "input": user_message})
    return (q or user_message).strip() or user_message


def retrieve_context(
    db: Session,
    *,
    org_id: str,
    question: str,
    top_k: int = 5,
) -> tuple[str, List[str]]:
    """
    Pinecone search (namespace=org_id) -> chunk ids -> fetch content from Postgres -> context string.
    Returns (context, match_ids).
    """
    index = get_index()
    results = index.search(
        namespace=org_id,
        query={"inputs": {"text": question}, "top_k": top_k},
        fields=["document_id", "chunk_index", "filename"],
    )

    hits = results["result"]["hits"]
    match_ids = [h["_id"] for h in hits]

    if not match_ids:
        return ("", [])

    placeholders = ", ".join([f":id{i}" for i in range(len(match_ids))])
    params: Dict[str, Any] = {"org_id": org_id, **{f"id{i}": match_ids[i] for i in range(len(match_ids))}}

    rows = db.execute(
        text(f"""
            SELECT id, content
            FROM chunks
            WHERE org_id = CAST(:org_id AS uuid)
            AND id IN ({placeholders})
        """),
        params,
    ).mappings().all()

    content_by_id = {str(r["id"]): r["content"] for r in rows}
    context = "\n\n---\n\n".join(content_by_id[i] for i in match_ids if i in content_by_id)
    return (context, match_ids)


def build_effective_system_prompt(
    *,
    org_system_prompt: Optional[str],
) -> str:
    org_system = (org_system_prompt or "").strip()
    if not org_system:
        return QA_SYSTEM_PROMPT
    return (org_system + "\n\n" + QA_SYSTEM_PROMPT).strip()


def generate_answer(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    user_message: str,
    lc_history: List[Tuple[str, str]],
    context: str,
) -> str:
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    chain = qa_prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "input": user_message,
        "chat_history": lc_history,
    })

    return (answer or "").strip()


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


@dataclass
class RagResult:
    standalone_question: str
    match_ids: List[str]
    context: str
    answer: str
