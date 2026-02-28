# rag/generator.py
from __future__ import annotations

import re
from typing import Dict, List, Literal, Sequence, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import rag.prompts as prompts


Role = Literal["user", "assistant"]
HistoryItem = Dict[str, str]  # {"role": "...", "content": "..."}


def build_llm(*, temperature: float, max_new_tokens: int = 512) -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=endpoint)


def history_to_lc(history: Sequence[HistoryItem]) -> List[Tuple[str, str]]:
    """Convert [{"role": "user"/"assistant", "content": "..."}] -> [("human", "..."), ("ai", "...")]."""
    out: List[Tuple[str, str]] = []
    for m in history:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append(("human", content) if role == "user" else ("ai", content))
    return out


def contextualize_question(
    llm: ChatHuggingFace,
    *,
    user_message: str,
    lc_history: List[Tuple[str, str]],
) -> str:
    if not lc_history:
        return user_message

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.contextualize_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    q = chain.invoke({"chat_history": lc_history, "input": user_message})
    return (q or user_message).strip() or user_message


def generate_answer(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    user_message: str,
    lc_history: List[Tuple[str, str]],
) -> str:
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = qa_prompt | llm | StrOutputParser()

    answer = chain.invoke({"input": user_message, "chat_history": lc_history})
    return (answer or "").strip()


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()