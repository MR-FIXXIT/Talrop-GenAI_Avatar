# rag/retriever.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pinecone_client import get_index
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Models loaded once at server startup ───────────────────────────────────────
#
# _EMBEDDING_MODEL : bi-encoder for fast dense retrieval        (~90 MB)
# _CROSS_ENCODER   : cross-encoder for precise local reranking  (~85 MB)
#
# Replacing the Cohere rerank API call with a local cross-encoder saves
# ~300-800 ms per request with zero network latency.
# Both models are thread-safe for read-only forward passes.
#
_EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_CROSS_ENCODER   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    page_number: int
    file_name: str


# ── Low-level building blocks (used by the parallel retrieval path) ────────────

def embed_query(question: str) -> List[float]:
    """
    Encode a single question into a dense vector.

    Safe to call from a ThreadPoolExecutor — SentenceTransformer inference
    is thread-safe for read-only forward passes.

    Args:
        question: Text to embed.

    Returns:
        List[float]: Dense embedding vector.
    """
    print("Embedding Query....")
    # show_progress_bar=False — tqdm adds overhead for single-sentence encoding
    return _EMBEDDING_MODEL.encode(question, show_progress_bar=False).tolist()


def pinecone_query(
    org_id: str,
    query_embedding: List[float],
    top_k: int = 20,
) -> List[RetrievedChunk]:
    """
    Query Pinecone with a pre-computed embedding and return raw (un-reranked) chunks.

    Args:
        org_id:          Pinecone namespace (tenant identifier).
        query_embedding: Dense query vector from embed_query().
        top_k:           Number of candidates to retrieve before reranking.

    Returns:
        List[RetrievedChunk]: Matched chunks ordered by vector similarity.
    """
    index = get_index()
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=org_id,
    )
    print("Retrieved records from vector store")

    matches = result["matches"]
    print(f"Length of matches : {len(matches)}")

    chunks: List[RetrievedChunk] = []
    for m in matches:
        metadata = m.get("metadata", {})
        chunks.append(
            RetrievedChunk(
                chunk_id=m.get("id", ""),
                text=metadata.get("text", ""),
                score=float(m.get("score", 0.0)),
                page_number=metadata.get("page_number", 0),
                file_name=metadata.get("filename", ""),
            )
        )
    return chunks


def rerank_chunks(
    question: str,
    chunks: List[RetrievedChunk],
    rerank_top_k: int = 3,
) -> List[RetrievedChunk]:
    """
    Re-rank candidate chunks using a local cross-encoder (no network call).

    The cross-encoder scores each (query, passage) pair jointly, giving
    much better relevance signals than bi-encoder cosine similarity alone.
    Running locally eliminates the Cohere API round-trip (~300-800 ms).

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - ~85 MB, fast CPU inference (~10-40 ms for 20 pairs)
      - Trained on MS MARCO passage ranking

    Args:
        question:     The search query.
        chunks:       Candidate chunks from pinecone_query().
        rerank_top_k: Number of top chunks to return after scoring.

    Returns:
        List[RetrievedChunk]: Top-k chunks ordered by cross-encoder score.
    """
    if not chunks:
        return []

    # Build (query, passage) pairs for joint scoring
    pairs = [(question, c.text or "") for c in chunks]

    # predict() returns a numpy array of relevance scores (higher = more relevant)
    scores = _CROSS_ENCODER.predict(pairs, show_progress_bar=False)

    # Sort descending by score and keep top-k
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in ranked[:rerank_top_k]]

    print(f"Reranked {len(chunks)} chunks -> kept top {len(top_chunks)} (local cross-encoder)")
    return top_chunks


# ── High-level convenience wrapper (backward-compatible) ──────────────────────

def retrieve_context(
    *,
    org_id: str,
    question: str,
    top_k: int = 20,        # BEFORE rerank
    min_score: float = 0.3,
    rerank_top_k: int = 3,  # AFTER rerank
) -> List[RetrievedChunk]:
    """
    Full retrieval pipeline: embed → Pinecone query → local cross-encoder rerank.

    Prefer calling embed_query / pinecone_query / rerank_chunks directly
    when you want to parallelise embedding with an upstream LLM call.
    """
    embedding = embed_query(question)
    raw_chunks = pinecone_query(org_id, embedding, top_k=top_k)

    if not raw_chunks:
        return []

    return rerank_chunks(question=question, chunks=raw_chunks, rerank_top_k=rerank_top_k)


if __name__ == "__main__":
    context_retrieved = retrieve_context(
        org_id="82142c62-9820-4a30-95d6-2f337cc7c2e5",
        question="Explain linear regression",
    )

    for i, c in enumerate(context_retrieved, start=1):
        print(f"Retrieved Context {i}")
        print(f"Score : {c.score}")
        print(f"Page number : {c.page_number}")
        print(f"File name : {c.file_name}")
        print(f"Context : {c.text}")
        print("============================")
