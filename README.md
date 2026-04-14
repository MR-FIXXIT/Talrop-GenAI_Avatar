# Conversational Avatar — GenAI RAG Backend

A multi-tenant, retrieval-augmented generation (RAG) API backend built with **FastAPI**, **Pinecone**, and **Groq (Qwen3-32B)**. Supports document ingestion, web scraping, and low-latency grounded chat with per-step pipeline profiling.

---

## Features

- **Multi-tenant RAG chat** — Namespace-isolated vector retrieval per organisation
- **Optimised latency pipeline** — Parallel embedding + question contextualisation; local cross-encoder reranking (no Cohere API round-trip)
- **Grounded answer generation** — Two-pass LLM pipeline: fact extraction → faithful answer synthesis (Qwen3-32B via Groq, thinking disabled)
- **Document ingestion** — Upload and chunk documents into Pinecone with metadata
- **Web scraping** — Ingest content from URLs into the vector store
- **Organisation & API-key management** — Full CRUD via REST API
- **pyinstrument profiling** — Per-request call-stack HTML profiles saved under `profiles/`
- **FastAPI + SQLAlchemy** — PostgreSQL-backed with automatic schema creation on startup

---

## Project Structure

```
.
├── main.py                    # FastAPI app entry point
├── auth.py                    # Authentication utilities
├── db.py                      # SQLAlchemy engine & session
├── pinecone_client.py         # Pinecone index client (singleton)
├── pinecone_setup.py          # Pinecone index initialisation
├── requirements.txt           # Python dependencies
│
├── models/                    # SQLAlchemy ORM models
│   ├── organization.py        #   Organisation model
│   ├── api_key.py             #   API key model
│   └── avatar_setting.py      #   Avatar settings model
│
├── routes/                    # FastAPI routers
│   ├── health.py              #   GET /health
│   ├── orgs.py                #   Organisation CRUD
│   ├── keys.py                #   API key management
│   ├── uploads.py             #   Document upload
│   ├── chat.py                #   Chat endpoint
│   └── scrape.py              #   Web scraping ingest
│
├── rag/                       # RAG pipeline
│   ├── chat_rag.py            #   Orchestrator — full pipeline with timing
│   ├── retriever.py           #   embed_query, pinecone_query, rerank_chunks
│   ├── generator.py           #   build_llm, contextualize_question, generate_faithful_answer
│   ├── ingestion.py           #   Document chunking & Pinecone upsert
│   ├── scraping.py            #   URL scrape + ingest
│   ├── textbook_chunker.py    #   Specialised textbook chunking strategy
│   └── prompts/               #   Prompt builders
│
├── profiles/                  # pyinstrument HTML profiles (auto-generated)
└── eval/                      # Evaluation scripts (deepeval)
```

---

## RAG Pipeline

Each chat request runs the following steps (with per-step wall-clock timing printed to stdout):

| Step | Description |
|------|-------------|
| 1 | Build 3 right-sized LLM instances (`thinking=False`) — ctx/extract/answer |
| 2 | Convert chat history to LangChain format |
| 3+4 *(parallel)* | Contextualise question (LLM, ~1–3 s) **∥** Embed original message (local, ~100 ms) |
| 4b | Re-embed if the question was rewritten; skip if unchanged |
| 5 | Pinecone vector query (`top_k=20`) |
| 6 | Local CrossEncoder rerank → keep top 3 (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| 7 | Format labelled context + build system prompts |
| 8 | Extract supported facts from context (LLM, 800 tok budget) |
| 9 | Generate faithful answer in one pass (LLM, 1 500 tok budget) |
| 10 | Normalize / strip `<think>` blocks from final answer |

> A full pyinstrument HTML profile is saved to `profiles/chat_profile_<timestamp>.html` after every request.

---

## Models Used

| Role | Model | Provider |
|------|-------|----------|
| LLM (all steps) | `qwen/qwen3-32b` | Groq |
| Query embedding | `all-MiniLM-L6-v2` | Sentence-Transformers (local) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sentence-Transformers (local) |

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Talrop-GenAI_Avatar
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string, e.g. `postgresql+psycopg://user:pass@localhost:5432/rag_saas` |
| `ADMIN_BOOTSTRAP_TOKEN` | Random secret used to seed admin credentials |
| `PINECONE_API_KEY` | API key from [pinecone.io](https://www.pinecone.io/) |
| `PINECONE_INDEX` | Name of your Pinecone index |
| `PINECONE_CLOUD` | Cloud provider for the index (e.g. `aws`) |
| `PINECONE_REGION` | Region for the index (e.g. `us-east-1`) |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com/) |
| `COHERE_API_KEY` | Cohere key (legacy — no longer used in the hot path) |

### 5. Initialise Pinecone

```bash
python pinecone_setup.py
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/orgs/` | Create organisation |
| `GET` | `/orgs/` | List organisations |
| `POST` | `/keys/` | Create API key for an org |
| `GET` | `/keys/` | List API keys |
| `POST` | `/uploads/` | Upload a document for ingestion |
| `POST` | `/scrape/` | Scrape a URL and ingest into vector store |
| `POST` | `/chat/` | Send a chat message and receive a RAG-grounded answer |

---

## Development Notes

- **Thread safety**: Both `SentenceTransformer` and `CrossEncoder` are loaded once at startup and are safe for concurrent read-only forward passes.
- **LLM token budgets**: Three separate `ChatGroq` instances are created per request with tight `max_tokens` limits (300 / 800 / 1 500) to prevent runaway generation cost.
- **Thinking disabled**: `reasoning_effort="none"` is set on all Groq calls to prevent Qwen3's `<think>` chains from silently consuming the token budget (previously causing 35–50 s steps).
- **Namespace isolation**: Each organisation's documents are stored in a dedicated Pinecone namespace (`org_id`), ensuring strict tenant separation.
