# rag/youtube_ingestion.py
from __future__ import annotations

import uuid
from typing import Any, List, Optional
from dataclasses import dataclass

from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_client import get_index
from rag.textbook_chunker import preprocess_text
from sentence_transformers import SentenceTransformer

# Load once (global)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@dataclass
class YouTubeIngestResult:
    document_id: str
    video_title: str
    url: str

def fetch_youtube_transcript(url: str) -> List[Any]:
    """
    Fetch transcript using LangChain's YoutubeLoader.
    """
    # Try with video info first, if it fails, try without
    try:
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=True,
            language=["en", "en-US", "en-GB"],
            translation="en"
        )
        return loader.load()
    except Exception:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            return loader.load()
        except Exception as e:
            raise Exception(f"Failed to fetch YouTube transcript: {e}")

def ingest_youtube_video_and_index(
    *,
    org_id: str,
    url: str,
    num_sentence_chunk_size: int = 10,
    min_token_length: int = 30
) -> YouTubeIngestResult:
    """
    Fetches transcript, chunks it, embeds, and upserts to Pinecone.
    """
    docs = fetch_youtube_transcript(url)
    if not docs:
        raise ValueError(f"No transcript found for video: {url}")

    # For standard YoutubeLoader, it usually returns one main document with the full transcript
    # and video info in metadata.
    full_text = docs[0].page_content
    metadata = docs[0].metadata
    video_title = metadata.get("title", "YouTube Video")
    
    doc_uuid = str(uuid.uuid4())

    # ---- chunking ----
    # Transcripts often lack punctuation, so sentence-based splitting might fail.
    # We use RecursiveCharacterTextSplitter for better results.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        raise ValueError("No valid chunks were produced from the transcription")

    index = get_index()

    texts = []
    metadatas = []
    ids = []

    # ---- prepare data ----
    for i, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        chunk_id = str(uuid.uuid4())

        texts.append(chunk_text)
        ids.append(chunk_id)
        
        # Merge original video metadata with chunk metadata
        chunk_meta = {
            "text": chunk_text,
            "document_id": doc_uuid,
            "chunk_index": i,
            "filename": f"youtube_{url.split('v=')[-1]}",
            "video_title": video_title,
            "source_url": url,
            "author": metadata.get("author", "Unknown"),
            "page_number": 1, # Default for text
        }
        metadatas.append(chunk_meta)

    if not texts:
        raise ValueError("No valid chunks after filtering")

    # ---- generate embeddings ----
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # ---- upsert to Pinecone ----
    vectors = []
    for i in range(len(texts)):
        vectors.append({
            "id": ids[i],
            "values": embeddings[i].tolist(),
            "metadata": metadatas[i],
        })

    from rag.ingestion import batch_list
    for batch in batch_list(vectors, 20):
        index.upsert(namespace=org_id, vectors=batch)

    return YouTubeIngestResult(
        document_id=doc_uuid,
        video_title=video_title,
        url=url
    )
