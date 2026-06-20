# Simple-RAG-Node

A minimal Retrieval-Augmented Generation pipeline in Node.js — PDF ingestion, vector search, and local LLM Q&A. Runs entirely locally with no API costs.

Node.js counterpart of [Simple-RAG](../Simple-RAG) (Python version).

## How It Works

1. **Extract** — pulls raw text from each PDF using `pdf-parse`
2. **Chunk** — splits text into overlapping segments (500 chars with 100 char overlap)
3. **Embed** — converts chunks into vectors using `@huggingface/transformers` (`all-MiniLM-L6-v2`, same model as the Python version)
4. **Store** — saves vectors + text + metadata in Qdrant for similarity search
5. **Query** — retrieves the most relevant chunks and passes them to a local LLM (Ollama + `llama3.2`) for answering questions

## Prerequisites

- Node.js 18+
- [Docker](https://docs.docker.com/get-docker/)
- [Ollama](https://ollama.com/) (free, runs locally)

## Setup

### 1. Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 2. Install and start Ollama

Download from [ollama.com](https://ollama.com/), then pull a model:

```bash
ollama pull llama3.2
```

### 3. Install dependencies

```bash
npm install
```

## Usage

### Ingest PDFs

```bash
# Single file
node ingest.js sample-data/source/document.pdf

# All PDFs at once (use shell glob)
node ingest.js sample-data/source/*.pdf

# Wipe the collection and re-ingest from scratch
node ingest.js sample-data/source/*.pdf --reset
```

### Query — Ask questions

```bash
node query.js ask "Who signed the documents?"
```

### Query — Extract structured fields

Extracts fields from each document and returns deduplicated JSON (one entry per source document):

```bash
node query.js extract "signature, date, email, IP"
```

Example output:

```json
[
  {
    "signature": "John Doe (Electronically signed)",
    "date": "1/15/2025",
    "email": "john.doe@gmail.com",
    "IP": "192.168.1.100",
    "_source": "Company_Handbook.pdf"
  },
  {
    "signature": "John Doe (Electronically signed)",
    "date": "1/15/2025",
    "email": "john.doe@gmail.com",
    "IP": "192.168.1.100",
    "_source": "Internal_Regulation.pdf"
  }
]
```

## Configuration

All settings live in `config.js`:

| Setting | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `Xenova/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBED_DIM` | `384` | Vector dimension (must match the model) |
| `QDRANT_HOST` | `http://localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION` | `simple_rag_node` | Qdrant collection name |
| `OLLAMA_MODEL` | `llama3.2` | Local LLM model for Q&A |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

## Project Structure

```
Simple-RAG-Node/
├── config.js          # All tunable constants
├── ingest.js          # PDF ingestion pipeline
├── query.js           # Q&A and structured field extraction
├── package.json       # Node dependencies
└── sample-data/
    └── source/        # Place PDF files here
```

## Python vs Node Comparison

| Component | Python | Node.js |
|---|---|---|
| PDF extraction | `pypdf` | `pdf-parse` |
| Embeddings | `sentence-transformers` | `@huggingface/transformers` |
| Vector DB client | `qdrant-client` | `@qdrant/js-client-rest` |
| LLM | `ollama` | `ollama` |
| Embedding model | `all-MiniLM-L6-v2` | `Xenova/all-MiniLM-L6-v2` |
