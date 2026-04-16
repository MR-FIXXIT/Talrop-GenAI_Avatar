#rag/chat_rag.py
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

from pyinstrument import Profiler

from rag.retriever import (
    RetrievedChunk,
    embed_query,
    pinecone_query,
    rerank_chunks,
)

from rag.generator import (
    HistoryItem,
    build_llm,
    history_to_lc,
    contextualize_question,
    generate_multi_queries,
    extract_supported_facts,
    generate_faithful_answer,   # replaces generate_answer_from_facts + revise_answer_for_faithfulness
    normalize_final_answer,
)

import rag.prompts as prompts



# Directory where HTML profiles are saved
_PROFILE_DIR = os.path.join(os.path.dirname(__file__), "..", "profiles")


@dataclass
class StepTiming:
    name: str
    elapsed: float  # seconds


def _print_timing_table(timings: List[StepTiming], total: float) -> None:
    """Print a formatted per-step timing table to stdout."""
    col_w = 38
    sep = "-" * (col_w + 22)
    print("\n" + sep)
    print(f"  {'CHAT PIPELINE TIMING':^{col_w + 18}}")
    print(sep)
    print(f"  {'Step':<{col_w}}  {'Time (s)':>8}  {'Share':>6}")
    print(sep)
    for t in timings:
        share = (t.elapsed / total * 100) if total > 0 else 0
        print(f"  {t.name:<{col_w}}  {t.elapsed:>8.3f}  {share:>5.1f}%")
    print(sep)
    print(f"  {'TOTAL':<{col_w}}  {total:>8.3f}  100.0%")
    print(sep + "\n")


def _save_profile(profiler: Profiler) -> str:
    """Persist the pyinstrument HTML report and return the file path."""
    os.makedirs(_PROFILE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_PROFILE_DIR, f"chat_profile_{stamp}.html")
    html = profiler.output_html(timeline=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(path)

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
    tone: Optional[str] = None,
    temperature: float,
    top_k: int = 20,
    min_score: float = 0.1,
) -> RagResult:
    """
    Orchestrates the full RAG pipeline with per-step timing and pyinstrument profiling.

    Steps:
      1)  Build 3 LLM instances (right-sized token budgets, thinking disabled)
      2)  Convert history to LangChain format
      3+4) Parallel: contextualize question (LLM) ∥ embed original message
      4b) Re-embed if question was rewritten, else reuse pre-computed embedding
      5)  Pinecone vector query
      6)  Local CrossEncoder rerank
      7)  Format context + build fact-extraction prompt
      8)  Extract supported facts (LLM)
      9)  Generate faithful answer in one pass (LLM)
      10) Normalize final answer

    A per-step timing table is printed to stdout after every request and a
    full pyinstrument HTML profile is saved under profiles/.
    """
    timings: List[StepTiming] = []
    pipeline_start = time.perf_counter()

    profiler = Profiler()
    profiler.start()

    #   llm_ctx     — 300 tokens   : rewrite question from history (~50 tokens out)
    #   llm_multi   — 500 tokens   : translate into 5 variations (~200 tokens out)
    #   llm_extract — 800 tokens   : extract supported facts     (~300 tokens out)
    #   llm_answer  — 1500 tokens  : final faithful answer       (~500-800 tokens out)
    t0 = time.perf_counter()
    llm_ctx     = build_llm(temperature=temperature, max_new_tokens=300,  thinking=False)
    llm_multi   = build_llm(temperature=temperature, max_new_tokens=500,  thinking=False)
    llm_extract = build_llm(temperature=temperature, max_new_tokens=800,  thinking=False)
    llm_answer  = build_llm(temperature=temperature, max_new_tokens=1500, thinking=False)
    timings.append(StepTiming("1. build_llm (4 instances, thinking=off)", time.perf_counter() - t0))

    # ── Step 2: Convert history to LangChain format ────────────────────────
    t0 = time.perf_counter()
    lc_history = history_to_lc(history)
    timings.append(StepTiming("2. history_to_lc", time.perf_counter() - t0))

    # ── Steps 3 & 4 (PARALLEL): contextualize question ∥ embed original msg ──
    #
    # contextualize_question makes an LLM call (~1-3 s when history exists).
    # embed_query runs the local SentenceTransformer (~50-150 ms).
    # Both are submitted at the same time; the embedding finishes first and
    # waits cheaply while the LLM rewrites the question.
    #
    # After both futures resolve:
    #   • If question unchanged  → use the pre-computed embedding (free).
    #   • If question was rewritten → re-embed the new question (~100 ms).
    #     The model is cached in memory so this is fast.
    parallel_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_q:   Future[str]        = pool.submit(
            contextualize_question, llm_ctx,
            user_message=user_message,
            lc_history=lc_history,
        )
        future_emb: Future[List[float]] = pool.submit(
            embed_query, user_message
        )
        # Block until both complete (LLM will almost always finish last)
        standalone_question: str        = future_q.result()
        orig_embedding:      List[float] = future_emb.result()

    parallel_elapsed = time.perf_counter() - parallel_start
    timings.append(StepTiming(
        "3+4. contextualize_question ∥ embed_query (parallel)",
        parallel_elapsed,
    ))

    # ── Step 4.5: Generate multiple queries (Multi-Query Translation) ──────────
    t0 = time.perf_counter()
    multi_queries = generate_multi_queries(llm_multi, question=standalone_question)
    timings.append(StepTiming(f"4.5 generate_multi_queries ({len(multi_queries)} queries)", time.perf_counter() - t0))

    # ── Step 5: Parallel Pinecone retrieval for all queries ───────────────────
    # We retrieve top_k for EACH variation, then deduplicate by chunk_id.
    t0 = time.perf_counter()
    all_raw_chunks: List[RetrievedChunk] = []
    
    with ThreadPoolExecutor(max_workers=min(5, len(multi_queries))) as pool:
        # 1. Embed all variations
        print(f"Embedding {len(multi_queries)} queries...")
        embeddings = list(pool.map(embed_query, multi_queries))
        
        # 2. Query Pinecone for each variation
        print(f"Querying Pinecone for {len(multi_queries)} variations...")
        future_results = [
            pool.submit(pinecone_query, org_id, emb, top_k=top_k) 
            for emb in embeddings
        ]
        
        for f in future_results:
            all_raw_chunks.extend(f.result())

    # ── Step 5.5: Deduplicate by chunk_id ────────────────────────────────────
    unique_chunks_dict = {c.chunk_id: c for c in all_raw_chunks if c.chunk_id}
    raw_chunks = list(unique_chunks_dict.values())
    
    # If Pinecone didn't return IDs (e.g. empty), fallback to all
    if not unique_chunks_dict:
        raw_chunks = all_raw_chunks

    timings.append(StepTiming(
        f"5. multi-query retrieval ({len(all_raw_chunks)} raw -> {len(raw_chunks)} unique)", 
        time.perf_counter() - t0
    ))

    # ── Step 6: CrossEncoder rerank ───────────────────────────────────────────
    t0 = time.perf_counter()
    # We rerank the unique set against the ORIGINAL standalone question
    chunks = rerank_chunks(question=standalone_question, chunks=raw_chunks)
    timings.append(StepTiming("6. rerank_chunks (CrossEncoder)", time.perf_counter() - t0))

    if not chunks:
        profiler.stop()
        total = time.perf_counter() - pipeline_start
        timings.append(StepTiming("  → early exit: no chunks retrieved", 0.0))
        _print_timing_table(timings, total)
        return RagResult(
            standalone_question=standalone_question,
            context="",
            answer=_NO_CONTEXT_ANSWER,
            supported_facts="NO_SUPPORT",
        )

    # ── Step 7: Format context ─────────────────────────────────────────────
    t0 = time.perf_counter()
    labeled_context = format_labeled_context(chunks)
    fact_extract_system_prompt = prompts.build_effective_fact_extract_system_prompt(
        org_system_prompt=org_system_prompt,
        context=labeled_context,
    )
    timings.append(StepTiming("7. format_context + build_prompts", time.perf_counter() - t0))

    # ── Step 8: Extract supported facts (LLM call) ─────────────────────────
    t0 = time.perf_counter()
    supported_facts = extract_supported_facts(
        llm_extract,
        system_prompt=fact_extract_system_prompt,
        question=standalone_question,
    )
    timings.append(StepTiming("8. extract_supported_facts (LLM, 800 tok)", time.perf_counter() - t0))

    # print(f"Supported Facts : {supported_facts}")

    if not supported_facts or supported_facts.strip() == "NO_SUPPORT":
        answer = _NO_CONTEXT_ANSWER
        timings.append(StepTiming("9. generate_faithful_answer (LLM, 1500 tok)", 0.0))
    else:
        faithful_answer_system_prompt = (
            prompts.build_effective_faithful_answer_system_prompt(
                org_system_prompt=org_system_prompt,
                tone=tone,
                context=labeled_context,
                supported_facts=supported_facts,
            )
        )

        # ── Step 9: Generate faithful answer in one pass (LLM call) ─────────
        # Replaces the old steps 9 + 10 (generate_answer_from_facts +
        # revise_answer_for_faithfulness).  Faithfulness rules are baked
        # into the prompt so no revision loop is needed.
        t0 = time.perf_counter()
        answer = generate_faithful_answer(
            llm_answer,
            system_prompt=faithful_answer_system_prompt,
            question=standalone_question,
        )
        timings.append(StepTiming("9. generate_faithful_answer (LLM, 1500 tok)", time.perf_counter() - t0))

    # ── Step 10: Normalize final answer ─────────────────────────────────────
    t0 = time.perf_counter()
    answer = normalize_final_answer(answer)
    timings.append(StepTiming("10. normalize_final_answer", time.perf_counter() - t0))

    # ── Stop profiler & report ─────────────────────────────────────────────
    profiler.stop()
    total = time.perf_counter() - pipeline_start

    _print_timing_table(timings, total)

    try:
        profile_path = _save_profile(profiler)
        print(f"[pyinstrument] Full call-stack profile saved → {profile_path}\n")
    except Exception as exc:  # never crash the request over profiling
        print(f"[pyinstrument] Could not save profile: {exc}")

    return RagResult(
        standalone_question=standalone_question,
        context=labeled_context,
        answer=answer,
        supported_facts=supported_facts,
    )