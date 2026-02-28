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
# Creates a LangChain chat model wrapper around a Hugging Face endpoint:
# 
# - Uses HuggingFaceEndpoint pointing to:
#   - repo_id="meta-llama/Llama-3.1-8B-Instruct"
# - Sets generation controls:
#   - temperature
#   - max_new_tokens
# - Wraps it in ChatHuggingFace so it can be used in LangChain prompt chains.
# 
# Outcome: a reusable llm object you can call via LangChain chains.
    
    return ChatHuggingFace(llm=endpoint)


def history_to_lc(history: Sequence[HistoryItem]) -> List[Tuple[str, str]]:
    # Convert [{"role": "user"/"assistant", "content": "..."}] -> [("human", "..."), ("ai", "...")].

    out: List[Tuple[str, str]] = []
    for m in history:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append(("human", content) if role == "user" else ("ai", content))
    return out


def contextualize_question(
# Goal: rewrite the current user message into a standalone question using prior chat context.
# 
# - If there’s no history, it returns user_message unchanged.
# - Otherwise it builds a prompt:
#   - system message from prompts.contextualize_system_prompt()
#   - inserts chat_history
#   - then includes the latest user input as {input}
# - Runs a LangChain “chain”:
#   - prompt | llm | StrOutputParser()
# - Returns the rewritten question (or falls back to original if output is empty).
# 
# Typical use: if the user says “What about that one?” the function rewrites it to something explicit like “What about King Faisal Air Base?” (depending on history).        

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
    
# This produces the final assistant answer.
# 
# - Constructs a chat prompt:
#   - ("system", system_prompt) (this is where you put your RAG rules, safety rules, formatting rules, “use only context”, etc.)
#   - includes chat_history
#   - includes latest user input
# - Executes the chain prompt | llm | StrOutputParser()
# - Returns the final model text (trimmed).
# 
# Note: This function itself does not do retrieval. It assumes you’ve already constructed system_prompt (often containing retrieved context).

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
    # Removes hidden reasoning tags if the model outputs them, e.g. <think>...</think>.
    return _THINK_RE.sub("", text).strip()