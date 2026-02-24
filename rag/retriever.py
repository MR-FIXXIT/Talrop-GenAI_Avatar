# rag/retriever.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text

from pinecone_client import get_index


def retrieve_context(
    db: Session,
    *,
    org_id: str,
    question: str,
    top_k: int = 5,
    min_score: float = 0.3,
) -> Tuple[str, List[str]]:
    """
    Pinecone search (namespace=org_id) -> chunk ids -> fetch content from Postgres -> context string.

    Returns:
        (context, match_ids)
    """
    index = get_index()
    results = index.search(
        namespace=org_id,
        query={"inputs": {"text": question}, "top_k": top_k},
        fields=["document_id", "chunk_index", "filename"],
    )

    hits = results.get("result", {}).get("hits", []) or []
    if not hits:
        return ("", [])

    best_score = float(hits[0].get("_score", 0.0) or 0.0)
    if best_score < min_score:
        return ("", [])

    match_ids = [h.get("_id") for h in hits if h.get("_id")]
    if not match_ids:
        return ("", [])

    placeholders = ", ".join([f":id{i}" for i in range(len(match_ids))])
    params: Dict[str, Any] = {
        "org_id": org_id,
        **{f"id{i}": match_ids[i] for i in range(len(match_ids))},
    }

    rows = (
        db.execute(
            text(
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

    content_by_id = {str(r["id"]): r["content"] for r in rows}
    context = "\n\n---\n\n".join(content_by_id[i] for i in match_ids if i in content_by_id)

    return (context, match_ids)