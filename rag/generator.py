# rag/generator.py

from __future__ import annotations

import re
from typing import Dict, List, Literal, Sequence, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import rag.prompts as prompts


Role = Literal["user", "assistant"]
HistoryItem = Dict[str, str]

_NO_CONTEXT_ANSWER = "Answer: I don't know based on the provided context."

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_llm(*, temperature: float, max_new_tokens: int) -> ChatHuggingFace:
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
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        repetition_penalty=1.05,
        do_sample=False if temperature == 0 else True,
    )
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


def _escape_lc_braces(text: str) -> str:
    # Prevent LangChain from treating any {..} in retrieved context/org prompt as template variables
    return text.replace("{", "{{").replace("}", "}}")


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def contextualize_question(
    llm: ChatHuggingFace,
    *,
    user_message: str,
    lc_history: List[Tuple[str, str]],
) -> str:
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

    if not lc_history:
        return user_message.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.contextualize_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    q = chain.invoke({"chat_history": lc_history, "input": user_message})
    q = strip_think(q or "")
    return q.strip() or user_message.strip()


def extract_supported_facts(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    question: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    facts = chain.invoke({"input": question})
    return strip_think(facts or "").strip()


def generate_answer_from_facts(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    question: str,
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"input": question})
    return strip_think(answer or "").strip()


def revise_answer_for_faithfulness(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    question: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    revised = chain.invoke({"input": question})
    revised = strip_think(revised or "").strip()
    return revised or _NO_CONTEXT_ANSWER


def normalize_final_answer(text: str) -> str:
    text = strip_think(text or "").strip()
    return text or _NO_CONTEXT_ANSWER