

"""Preprocessing and chunking pipeline for RAG.

What it does:
1. Reads text from a PDF with PyMuPDF or accepts raw text.
2. Cleans text.
3. Splits into sentences with spaCy's sentencizer.
4. Groups sentences into fixed-size sentence chunks.
5. Computes chunk statistics.
6. Filters out very short chunks.
7. Returns the final chunk DataFrame for ingestion into a RAG pipeline.

Install requirements first:
    pip install pymupdf pandas spacy tqdm

If spaCy is not installed in your environment, also install it:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re

import pymupdf
import pandas as pd
from spacy.lang.en import English

from pathlib import Path



# Sentence chunk size and minimum chunk length filter.
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30


def text_formatter(text: str) -> str:
    """Clean extracted text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf_from_bytes(pdf_bytes: bytes) -> list[dict]:
    """Read a PDF from memory and extract text/statistics for each page."""
    print("Pdf Reading....")
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages_and_texts: list[dict] = []

    for page_number, page in enumerate(doc):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append(
            {
                "page_number": page_number + 1,
                "page_char_count": len(text),   
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
            }
        )

    doc.close()
    return pages_and_texts


def add_sentence_data(pages_and_texts: list[dict], nlp: English) -> None:
    """Add sentence lists and sentence counts to each page item."""

    for item in pages_and_texts:
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])


def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    """Split a list into sublists of size `slice_size`."""
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]

def add_sentence_chunks(
    pages_and_texts: list[dict],
    num_sentence_chunk_size: int = NUM_SENTENCE_CHUNK_SIZE,
) -> None:
    """Group page sentences into fixed-size chunks."""
    for item in pages_and_texts:
        item["sentence_chunks"] = split_list(
            input_list=item["sentences"],
            slice_size=num_sentence_chunk_size,
        )
        item["num_chunks"] = len(item["sentence_chunks"])



def build_chunk_records(pages_and_texts: list[dict]) -> list[dict]:
    """Flatten sentence chunks into chunk records with metadata and stats."""
    pages_and_chunks: list[dict] = []

    for item in pages_and_texts:
        for sentence_chunk in item["sentence_chunks"]:
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(
                r"\.([A-Z])",
                r". \1",
                joined_sentence_chunk,
            )  # ".A" -> ". A"

            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": joined_sentence_chunk,
                "chunk_char_count": len(joined_sentence_chunk),
                "chunk_word_count": len(joined_sentence_chunk.split(" ")),
                "chunk_token_count": len(joined_sentence_chunk) / 4,  # ~4 chars/token
            }
            pages_and_chunks.append(chunk_dict)
    
    print(f"length pages_and_chunks : {len(pages_and_chunks)}")

    return pages_and_chunks



def filter_short_chunks(
    chunks_df: pd.DataFrame,
    min_token_length: int = MIN_TOKEN_LENGTH,
) -> list[dict]:
    """Keep only chunks above the minimum approximate token length."""
    return chunks_df[chunks_df["chunk_token_count"] > min_token_length].to_dict(
        orient="records"
    )



def preprocess_pdf(
    pdf_bytes: bytes, 
    num_sentence_chunk_size: int = NUM_SENTENCE_CHUNK_SIZE,
    min_token_length: int = MIN_TOKEN_LENGTH,
) -> pd.DataFrame:
    """Run the full preprocessing + chunking pipeline and return final chunk DataFrame."""
    pages_and_texts = open_and_read_pdf_from_bytes(pdf_bytes)

    nlp = English()
    nlp.add_pipe("sentencizer")

    add_sentence_data(pages_and_texts, nlp)
    add_sentence_chunks(pages_and_texts, num_sentence_chunk_size=num_sentence_chunk_size)

    pages_and_chunks = build_chunk_records(pages_and_texts)
    chunks_df = pd.DataFrame(pages_and_chunks)
    filtered_chunks = filter_short_chunks(chunks_df, min_token_length=min_token_length)

    print(f"length filtered_chunks: {len(filtered_chunks)}")

    return pd.DataFrame(filtered_chunks)


def preprocess_text(
    text: str,
    num_sentence_chunk_size: int = NUM_SENTENCE_CHUNK_SIZE,
    min_token_length: int = MIN_TOKEN_LENGTH,
) -> pd.DataFrame:
    """Preprocess raw text string into chunks."""
    text = text_formatter(text)
    
    # We treat the whole text as "one page" for the sake of the existing pipeline
    pages_and_texts = [
        {
            "page_number": 1,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text,
        }
    ]

    nlp = English()
    nlp.add_pipe("sentencizer")

    add_sentence_data(pages_and_texts, nlp)
    add_sentence_chunks(pages_and_texts, num_sentence_chunk_size=num_sentence_chunk_size)

    pages_and_chunks = build_chunk_records(pages_and_texts)
    chunks_df = pd.DataFrame(pages_and_chunks)
    filtered_chunks = filter_short_chunks(chunks_df, min_token_length=min_token_length)

    print(f"length filtered_chunks: {len(filtered_chunks)}")

    return pd.DataFrame(filtered_chunks)





# if __name__ == "__main__":
#     pdf_path = Path("C:/Users/banuv/Desktop/wegnio/CUSAT/SEM 2/dip/Dip module 1.pdf")
#     with open(pdf_path, 'rb') as file:
#         pdf_bytes = file.read()

#     d =preprocess_textbook_pdf(pdf_bytes=pdf_bytes)
#     print(d.columns)
