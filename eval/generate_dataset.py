import argparse
import json
import os
import sys
import tempfile
from typing import List

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from sentence_transformers import SentenceTransformer

# Ensure we can import from the main project directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the existing scraping module safely
from rag.scraping import scrape_static_url
from rag.generator import build_llm

load_dotenv()

class GroqSynthesizerModel(DeepEvalBaseLLM):
    def __init__(self):
        # Using build_llm to fetch the configured Groq model (e.g. Qwen3-32b) without reasoning chains
        self._model = build_llm(temperature=0.3, max_new_tokens=4096, thinking=False)

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self._model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "ChatGroq"

class STEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_model(self):
        return self._model

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self):
        return "all-MiniLM-L6-v2"

def scrape_to_tempfile(url: str, temp_dir: str) -> str:
    """Scrapes a URL and saves its textual content to a local temp .txt file."""
    print(f"[*] Scraping URL: {url}")
    page = scrape_static_url(url)
    
    # Create a safe filename from the title or just fallback
    safe_name = "".join(c for c in page.title if c.isalnum() or c in " _-").strip()
    if not safe_name:
        safe_name = "scraped_page"
        
    file_path = os.path.join(temp_dir, f"{safe_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(page.text)
        
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eval questions from docs and websites using DeepEval.")
    parser.add_argument("--docs", nargs="*", default=[], help="Paths to local documents (e.g., .txt, .pdf).")
    parser.add_argument("--urls", nargs="*", default=[], help="URLs to scrape and use as source context.")
    parser.add_argument("--output", default="eval/generated_dataset.json", help="Path to save the output JSON.")
    parser.add_argument("--num-questions", type=int, default=30, help="Target number of total questions to generate.")
    
    args = parser.parse_args()

    document_paths: List[str] = [os.path.abspath(p) for p in args.docs]
    
    # We need a directory to temporarily store scraped website content as .txt
    # so that DeepEval's Synthesizer can read them.
    temp_dir = tempfile.mkdtemp(prefix="deepeval_scrape_")
    
    for url in args.urls:
        try:
            temp_path = scrape_to_tempfile(url, temp_dir)
            document_paths.append(temp_path)
        except Exception as e:
            print(f"[!] Failed to scrape {url}: {e}")

    if not document_paths:
        print("[!] No documents or URLs provided. Please provide --docs or --urls.")
        sys.exit(1)

    print(f"\n[*] Synthesizing from {len(document_paths)} total sources.")
    
    # Calculate questions per document to roughly hit the target total
    count_per_doc = max(1, args.num_questions // len(document_paths))

    print(f"[*] Initializing DeepEval Synthesizer...")
    print(f"[*] Attempting to generate ~{count_per_doc} questions per document.\n")
    
    custom_model = GroqSynthesizerModel()
    custom_embedder = STEmbeddingModel()
    synthesizer = Synthesizer(model=custom_model, embedder=custom_embedder)
    
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        max_goldens_per_context=count_per_doc,
        include_expected_output=True,
    )

    print(f"\n[*] Generated {len(goldens)} total goldens from Synthesizer.")

    # Format exactly to your JSON schema
    formatted_data = []
    for i, g in enumerate(goldens):
        # Extract metadata if available
        meta = getattr(g, "additional_metadata", {}) or {}
        evolution = meta.get("evolution_type", "general")
        
        formatted_data.append({
            "id": i + 1,
            "type": evolution,
            "difficulty": "medium", 
            "query": g.input,
            "expected_answer": g.expected_output,
        })

    # If the synthesizer slightly overshot the exact number, clip the results to the requested max
    if len(formatted_data) > args.num_questions:
        formatted_data = formatted_data[:args.num_questions]
        
    # Re-index
    for i, item in enumerate(formatted_data):
        item["id"] = i + 1

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"[*] Successfully saved {len(formatted_data)} questions to {args.output}\n")


if __name__ == "__main__":
    main()
