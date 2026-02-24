# rag/generator.py
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import re

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


QA_SYSTEM_PROMPT = """You must answer using ONLY the provided Context.
Do NOT use outside knowledge. Do NOT guess.

Rules:
- If the answer is not explicitly stated in Context, reply exactly: I don't know.
- If Context is empty, reply exactly: I don't know.
- Keep the answer concise and directly relevant to the question.

Context:
{context}
"""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""


Role = Literal["user", "assistant"]
HistoryItem = Dict[str, str]  # {"role": "...", "content": "..."}


def build_llm(*, temperature: float, max_new_tokens: int = 512) -> ChatHuggingFace:
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    q = chain.invoke({"chat_history": lc_history, "input": user_message})
    return (q or user_message).strip() or user_message


def build_effective_system_prompt(*, org_system_prompt: Optional[str]) -> str:
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
    if not context or not context.strip():
        return "I don't know."

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = qa_prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "input": user_message,
            "chat_history": lc_history,
        }
    )

    return (answer or "").strip()


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()