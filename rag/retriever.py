# rag/retriever.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cohere

from pinecone_client import get_index
from sentence_transformers import SentenceTransformer

import os

co = cohere.Client(os.environ["COHERE_API_KEY"])


@dataclass
class RetrievedChunk:
    text: str
    score: float
    page_number: int
    file_name: str


def retrieve_context(
    *,  
    org_id: str,
    question: str,
    top_k: int = 20,  # BEFORE rerank
    min_score: float = 0.3,
    rerank_top_k: int = 3,  # AFTER rerank
) -> List[RetrievedChunk]:

    index = get_index()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding Query....")
    query_embedding = embedding_model.encode(question, show_progress_bar=True).tolist()

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=org_id
    )
    print("Retrieved records from vector store")


    matches = result["matches"]
    print(f"Length of matches : {len(matches)}")
    retrieved_chunks: List[RetrievedChunk] = []

    for m in matches:

        score = float(m.get("score"))

        # filter
        # if score < min_score:
        #     continue

        chunk_text = m.get("metadata").get("text")
        page_number = m.get("metadata").get("page_number")
        file_name = m.get("metadata").get("filename")

        retrieved_chunks.append(
            RetrievedChunk(
                text=chunk_text,
                score=score,
                page_number=page_number,
                file_name=file_name,
            )
        )

    if not retrieved_chunks:
        return []

    docs = [c.text for c in retrieved_chunks]

    response = co.rerank(
        query=question,
        documents=docs,
        top_n=rerank_top_k
    )
    print("Reranked chunks")

    # map reranked results back
    reranked_chunks: List[RetrievedChunk] = []
    for r in response.results:
        original_chunk = retrieved_chunks[r.index]  
        reranked_chunks.append(original_chunk)

    return reranked_chunks


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
