from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from db import SessionLocal

# Your retriever (black box)
from rag.retriever import retrieve_context


# -----------------------------
# Dataset IO
# -----------------------------
@dataclass(frozen=True)
class EvalExample:
    qid: str
    question: str
    relevant_chunk_ids: List[str]


def load_jsonl(path: Path) -> List[EvalExample]:
    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("qid") or f"line_{line_no}")
            question = str(obj["question"])
            rel = obj.get("relevant_chunk_ids") or []
            if not isinstance(rel, list):
                raise ValueError(f"{path}:{line_no}: relevant_chunk_ids must be a list")
            examples.append(EvalExample(qid=qid, question=question, relevant_chunk_ids=[str(x) for x in rel]))
    return examples


# -----------------------------
# Deterministic IR Metrics
# -----------------------------
def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    rel_set = set(relevant)
    topk = list(retrieved[:k])
    if not topk:
        return 0.0
    hits = sum(1 for x in topk if x in rel_set)
    # fixed-k definition (penalizes returning fewer than k)
    return hits / float(k)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel_set = set(relevant)
    if not rel_set:
        # avoid dividing by 0; you can also choose to skip such examples
        return 0.0
    topk = set(retrieved[:k])
    hits = len(topk.intersection(rel_set))
    return hits / float(len(rel_set))


# -----------------------------
# Helper: split your context string into list[str]
# (so it can be fed to Ragas as contexts)
# -----------------------------
SEP = "\n\n---\n\n"


def context_to_contexts(context: str) -> List[str]:
    context = (context or "").strip()
    if not context:
        return []
    parts = [p.strip() for p in context.split(SEP)]
    return [p for p in parts if p]

def fetch_chunk_texts_by_ids(db: Session, org_id: str, ids: List[str]) -> Dict[str, str]:
    if not ids:
        return {}
    placeholders = ", ".join([f":id{i}" for i in range(len(ids))])
    params: Dict[str, Any] = {"org_id": org_id, **{f"id{i}": ids[i] for i in range(len(ids))}}

    rows = (
        db.execute(
            sql_text(
                f"""
                SELECT id, content
                FROM chunks
                WHERE org_id = CAST(:org_id AS uuid)
                  AND id IN ({placeholders})
                """
            ),
            params,
        )
        .mappings()
        .all()
    )
    return {str(r["id"]): r["content"] for r in rows}


# -----------------------------
# Ragas integration (handles v0.3.x vs v0.4+ import paths)
# -----------------------------
def ragas_metrics() -> Tuple[Any, Any]:
    """
    Returns metric objects (or callables) for context_precision / context_recall across Ragas versions.
    - v0.3.x: from ragas.metrics import context_precision, context_recall
    - v0.4+:  from ragas.metrics.collections import ContextPrecision, ContextRecall
    """
    try:
        # v0.3 style (functions/Metric instances)
        from ragas.metrics import context_precision, context_recall  # type: ignore
        return context_precision, context_recall
    except Exception:
        # v0.4+ style (classes)
        from ragas.metrics.collections import ContextPrecision, ContextRecall  # type: ignore
        return ContextPrecision(), ContextRecall()


def ragas_evaluate(*, questions: List[str], contexts: List[List[str]], references: List[str], llm=None, embeddings=None) -> pd.DataFrame:
    from datasets import Dataset
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "reference": references,   # <-- required by your ragas version
        }
    )
    from ragas import evaluate
    cp, cr = ragas_metrics()
    result = evaluate(dataset=dataset, metrics=[cp, cr], llm=llm, embeddings=embeddings)
    return result.to_pandas()


# -----------------------------
# Runner
# -----------------------------
def run_eval(
    *,
    db: Session,
    org_id: str,
    dataset_path: Path,
    k_values: List[int],
    min_score: float,
    out_csv: Optional[Path],
    use_ragas: bool,
) -> pd.DataFrame:
    examples = load_jsonl(dataset_path)

    rows: List[Dict[str, Any]] = []
    all_questions: List[str] = []
    all_contexts: List[List[str]] = []
    all_references: List[str] = []

    for ex in examples:
        context, match_ids = retrieve_context(
            db,
            org_id=org_id,
            question=ex.question,
            top_k=max(k_values),
            min_score=min_score,
        )

        row: Dict[str, Any] = {
            "qid": ex.qid,
            "question": ex.question,
            "min_score": min_score,
            "retrieved_count": len(match_ids),
            "relevant_count": len(ex.relevant_chunk_ids),
        }

        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(match_ids, ex.relevant_chunk_ids, k)
            row[f"recall@{k}"] = recall_at_k(match_ids, ex.relevant_chunk_ids, k)

        # for debugging
        row["retrieved_ids"] = json.dumps(match_ids)
        row["relevant_ids"] = json.dumps(ex.relevant_chunk_ids)

        rows.append(row)

        # collect for ragas
        if use_ragas:
            # retrieved contexts (already)
            all_questions.append(ex.question)
            all_contexts.append(context_to_contexts(context))

            # build reference text from the gold chunk ids
            gold_texts = fetch_chunk_texts_by_ids(db, org_id, ex.relevant_chunk_ids)
            reference = "\n\n---\n\n".join(
                gold_texts[i] for i in ex.relevant_chunk_ids if i in gold_texts
            )
            all_references.append(reference)

    df = pd.DataFrame(rows)

    if use_ragas:
        # llm/embeddings left as None by default; wire them in if you want model-based steps
        ragas_df = ragas_evaluate(
            questions=all_questions,
            contexts=all_contexts,
            references=all_references,
            llm=None,
            embeddings=None,
        )

        # merge ragas scores back (row order should align)
        # ragas_df typically includes columns like "context_precision", "context_recall"
        for col in ragas_df.columns:
            if col in ("question",):
                continue
            df[col] = ragas_df[col].values

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df


def summarize(df: pd.DataFrame, k_values: List[int]) -> None:
    cols = []
    for k in k_values:
        cols.extend([f"precision@{k}", f"recall@{k}"])
    # include ragas columns if present
    for c in ("context_precision", "context_recall"):
        if c in df.columns:
            cols.append(c)

    summary = df[cols].mean(numeric_only=True).to_dict()
    print(json.dumps(summary, indent=2))


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--org-id", required=True)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--k", default="1,3,5", help="comma-separated k values, e.g. 1,3,5,10")
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ragas", action="store_true", help="also compute ragas context_precision/context_recall")
    args = p.parse_args()

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    k_values = sorted(set(k_values))
    if not k_values:
        raise ValueError("Provide at least one k via --k")

    with SessionLocal() as db:
        df = run_eval(
            db=db,
            org_id=args.org_id,
            dataset_path=args.dataset,
            k_values=k_values,
            min_score=args.min_score,
            out_csv=args.out,
            use_ragas=args.ragas,
        )
        summarize(df, k_values)


if __name__ == "__main__":
    main()