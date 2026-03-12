# rag/retriever.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.orm import Session

from pinecone_client import get_index


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


def retrieve_context(
    db: Session,
    *,
    org_id: str,
    question: str,
    top_k: int = 5,
    min_score: float = 0.3,
    return_chunks: bool = False,
) -> List[RetrievedChunk] | tuple[str, List[str]]:
    """
    Pinecone search (namespace=org_id) -> chunk ids -> fetch content from Postgres.

    If return_chunks=True:
        returns List[RetrievedChunk]

    Else:
        returns (context, match_ids) for backward compatibility
    """

    index = get_index()

    results = index.search(
        namespace=org_id,
        query={"inputs": {"text": question}, "top_k": top_k},
        fields=["document_id", "chunk_index", "filename"]
    )

    hits = results.get("result", {}).get("hits", []) or []

    if not hits:
        return [] if return_chunks else ("", [])

    best_score = float(hits[0].get("_score", 0.0) or 0.0)
    if best_score < min_score:
        return [] if return_chunks else ("", [])

    # Keep score + rank order from Pinecone
    ranked_hits = []
    for h in hits:
        chunk_id = h.get("_id")
        if not chunk_id:
            continue

        score = float(h.get("_score", 0.0) or 0.0)

        # filter every hit, not just best hit
        if score < min_score:
            continue

        ranked_hits.append(
            {
                "chunk_id": str(chunk_id),
                "score": score,
            }
        )

    if not ranked_hits:
        return [] if return_chunks else ("", [])

    match_ids = [h["chunk_id"] for h in ranked_hits]

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

    content_by_id = {str(r["id"]): (r["content"] or "").strip() for r in rows}

    retrieved_chunks: List[RetrievedChunk] = []
    for h in ranked_hits:
        chunk_id = h["chunk_id"]
        chunk_text = content_by_id.get(chunk_id, "").strip()
        if not chunk_text:
            continue

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                score=h["score"],
            )
        )

    if return_chunks:
        return retrieved_chunks

    context = "\n\n---\n\n".join(chunk.text for chunk in retrieved_chunks)
    match_ids = [chunk.chunk_id for chunk in retrieved_chunks]
    return (context, match_ids)