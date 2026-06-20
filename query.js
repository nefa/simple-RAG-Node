/**
 * query.js — search the vector database and get answers from the LLM.
 *
 * Two modes:
 *   node query.js ask "What does the handbook say about remote work?"
 *   node query.js extract "signature, date, email, IP"
 */

import { pipeline } from "@huggingface/transformers";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Ollama } from "ollama";

import {
  EMBED_MODEL,
  QDRANT_HOST,
  QDRANT_PORT,
  COLLECTION,
  OLLAMA_MODEL,
  TOP_K,
} from "./config.js";

let embedder = null;

async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", EMBED_MODEL, {
      dtype: "fp32",
    });
  }
  return embedder;
}

function parseJSON(text) {
  const cleaned = text.replace(/^```(?:json)?\s*|\s*```$/g, "").trim();
  return JSON.parse(cleaned);
}

// ── Retrieve relevant chunks ─────────────────────────────────────────────────

async function retrieve(query, topK = TOP_K) {
  const client = new QdrantClient({
    url: `${QDRANT_HOST}:${QDRANT_PORT}`,
  });
  const embed = await getEmbedder();
  const output = await embed(query, { pooling: "mean", normalize: true });
  const queryVector = Array.from(output.data);

  const results = await client.query(COLLECTION, {
    query: queryVector,
    limit: topK,
    with_payload: true,
  });

  return results.points.map((hit) => hit.payload);
}

// ── Ask mode ─────────────────────────────────────────────────────────────────

async function ask(question) {
  const chunks = await retrieve(question);
  const context = chunks
    .map((c) => `[${c.source} p.${c.page}]\n${c.text}`)
    .join("\n\n---\n\n");

  const ollama = new Ollama();
  const response = await ollama.chat({
    model: OLLAMA_MODEL,
    messages: [
      {
        role: "system",
        content:
          "You are a helpful assistant. Answer the question using " +
          "only the provided context. If the answer is not in the " +
          "context, say so.",
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${question}`,
      },
    ],
  });

  return response.message.content;
}

// ── Extract mode ─────────────────────────────────────────────────────────────

async function extract(fields) {
  const fieldList = fields.split(",").map((f) => f.trim());
  const query = "electronic signature date email IP address";
  const chunks = await retrieve(query);

  const grouped = {};
  for (const chunk of chunks) {
    if (!grouped[chunk.source]) grouped[chunk.source] = [];
    grouped[chunk.source].push(chunk);
  }

  const ollama = new Ollama();
  const results = [];

  for (const [source, sourceChunks] of Object.entries(grouped)) {
    const context = sourceChunks.map((c) => c.text).join("\n\n");

    const response = await ollama.chat({
      model: OLLAMA_MODEL,
      messages: [
        {
          role: "system",
          content:
            "You are a data extraction assistant. Extract the requested " +
            "fields from the provided context. Return valid JSON only — " +
            "a single object with the requested fields. " +
            "Use null for missing fields.",
        },
        {
          role: "user",
          content:
            `Context:\n${context}\n\n` +
            `Extract these fields: ${JSON.stringify(fieldList)}\n` +
            "Return JSON only, no explanation.",
        },
      ],
    });

    let entry;
    try {
      entry = parseJSON(response.message.content);
    } catch {
      entry = { _raw: response.message.content };
    }
    entry._source = source;
    results.push(entry);
  }

  return JSON.stringify(results, null, 2);
}

// ── CLI entry point ──────────────────────────────────────────────────────────

const [mode, ...rest] = process.argv.slice(2);
const arg = rest.join(" ");

if (!mode || !arg) {
  console.log(
    'Usage:\n  node query.js ask "your question"\n  node query.js extract "field1, field2"'
  );
  process.exit(1);
}

if (mode === "ask") {
  console.log(await ask(arg));
} else if (mode === "extract") {
  console.log(await extract(arg));
} else {
  console.log(`Unknown mode '${mode}'. Use 'ask' or 'extract'.`);
  process.exit(1);
}
