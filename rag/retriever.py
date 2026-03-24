# rag/retriever.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pinecone_client import get_index


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
    top_k: int ,
    min_score: float = 0.3,
) -> List[RetrievedChunk]:

    index = get_index()

    results = index.search(
        namespace=org_id,
        query={"inputs": {"text": question}, "top_k": top_k},
        fields=["page_number", "filename", "text"]
    )

    hits = results.get("result", {}).get("hits", []) or []

    retrieved_chunks: List[RetrievedChunk] = []
    for h in hits:

        score = float(h.get("_score", 0.0))

        # filter every hit, not just best hit
        if score < min_score:
            break

        score = h["_score"]
        chunk_text = h.get("fields", {}).get("text", "")
        page_number = h.get("fields", {}).get("page_number")
        file_name = h.get("fields", {}).get("filename", "")

        retrieved_chunks.append(
            RetrievedChunk(
                text=chunk_text,
                score=score,
                page_number=page_number,
                file_name=file_name,
            )
        )

    return retrieved_chunks



if __name__ == "__main__":
    context_retrieved = retrieve_context(org_id="0e6dd009-b0ea-47b5-9083-85239f62d5ba",
                                         top_k=5,
                                         question="What is a fourier transform?") 
    
    for i,c in enumerate(context_retrieved, start=1):
        print(f"Retrieved Context {i}")
        print(f"Score : {c.score}")
        print(f"Page number : {c.page_number}")
        print(f"File name : {c.file_name}")
        print(f"Context : {c.text}")
        print("============================")