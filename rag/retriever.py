# rag/retriever.py
from __future__ import annotations

from ast import If
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


    # 1) Search Pinecone for similar chunks
    # - Searches only inside namespace=org_id (tenant isolation)
    # - question is used as the query text
    # - Returns top top_k matches (“hits”)
    # - Requests some metadata fields, but this function doesn’t use them—it mainly uses _id and _score.
    results = index.search(
        namespace=org_id,
        query={"inputs": {"text": question}, "top_k": top_k},
        fields=["document_id", "chunk_index", "filename"],
    )
    # ////////////////////////////////////////////////////

    # 2) Basic filtering: no hits or too low confidence
    # - If Pinecone returns nothing → no context
    # - If the best match score is below min_score (default 0.3) → treat as irrelevant, return empty context
    hits = results.get("result", {}).get("hits", []) or []
    if not hits:
        return ("", [])

    best_score = float(hits[0].get("_score", 0.0) or 0.0)
    if best_score < min_score:
        return ("", [])
    # ////////////////////////////////////////////////////
    

    # 3) Collect Pinecone match IDs (chunk IDs)
    # Your ingestion code upserted Pinecone records with:
    # - id = chunk_id (UUID used in Postgres chunks.id)
    # So Pinecone _id == Postgres chunks.id. This is the key join.
    match_ids = [h.get("_id") for h in hits if h.get("_id")]
    if not match_ids:
        return ("", [])
    # ////////////////////////////////////////////////////
    


    placeholders = ", ".join([f":id{i}" for i in range(len(match_ids))])
    params: Dict[str, Any] = {
        "org_id": org_id,
        **{f"id{i}": match_ids[i] for i in range(len(match_ids))},
    }


    # 4) Fetch chunk text from Postgres for those IDs
    # It dynamically builds a parameterized IN (...) query:
    # - Creates placeholders like :id0, :id1, :id2, ...
    # - Builds params dict including org_id and each idN
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
    # This ensures you only pull chunks belonging to the current org.
    # ////////////////////////////////////////////////////

    
    # 5) Build the final context string in ranking order
    # DB results might come back in any order, so it maps id → content
    # Then concatenates content in the same order Pinecone ranked them (match_ids order)
    # Separates chunks with ---
    content_by_id = {str(r["id"]): r["content"] for r in rows}
    context = "\n\n---\n\n".join(content_by_id[i] for i in match_ids if i in content_by_id)
    # ////////////////////////////////////////////////////


    return (context, match_ids)