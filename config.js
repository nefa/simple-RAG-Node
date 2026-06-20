/**
 * config.js — central place for all tunable constants.
 *
 * Change values here; nothing else needs editing to adjust behavior.
 */

// ── Embedding model ──────────────────────────────────────────────────────────
// HuggingFace Transformers.js model used to convert text chunks into vectors.
// "all-MiniLM-L6-v2" is fast and small (~90 MB). Same model as the Python version.
export const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
export const EMBED_DIM = 384;

// ── Qdrant (vector database) ─────────────────────────────────────────────────
// Run locally with: docker run -d -p 6333:6333 qdrant/qdrant
export const QDRANT_HOST = "http://localhost";
export const QDRANT_PORT = 6333;
export const COLLECTION = "simple_rag_node";

// ── LLM (Ollama — runs locally, no API costs) ───────────────────────────────
// Pull the model first: ollama pull llama3.2
export const OLLAMA_MODEL = "llama3.2";

// ── Chunking ─────────────────────────────────────────────────────────────────
export const CHUNK_SIZE = 500;
export const CHUNK_OVERLAP = 100;

// ── Retrieval ────────────────────────────────────────────────────────────────
export const TOP_K = 4;
