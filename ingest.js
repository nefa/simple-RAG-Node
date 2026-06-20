/**
 * ingest.js — load PDFs into the vector database.
 *
 * Usage:
 *   node ingest.js sample-data/source/document.pdf
 *   node ingest.js sample-data/source/*.pdf
 *   node ingest.js sample-data/source/*.pdf --reset
 */

import { readFileSync } from "fs";
import { basename } from "path";
import { randomUUID } from "crypto";
import pdf from "pdf-parse/lib/pdf-parse.js";
import { pipeline } from "@huggingface/transformers";
import { QdrantClient } from "@qdrant/js-client-rest";
import {
  EMBED_MODEL,
  EMBED_DIM,
  QDRANT_HOST,
  QDRANT_PORT,
  COLLECTION,
  CHUNK_SIZE,
  CHUNK_OVERLAP,
} from "./config.js";

// ── Step 1: PDF → raw text ───────────────────────────────────────────────────

async function extractPages(pdfPath) {
  const buffer = readFileSync(pdfPath);
  const data = await pdf(buffer);
  const text = data.text.trim();
  if (!text) return [];

  return [
    {
      text,
      page: 1,
      source: basename(pdfPath),
    },
  ];
}

// ── Step 2: raw text → overlapping chunks ────────────────────────────────────

function chunkPages(pages) {
  const chunks = [];
  for (const page of pages) {
    const { text } = page;
    let start = 0;
    while (start < text.length) {
      const chunkText = text.slice(start, start + CHUNK_SIZE).trim();
      if (chunkText) {
        chunks.push({
          text: chunkText,
          page: page.page,
          source: page.source,
        });
      }
      start += CHUNK_SIZE - CHUNK_OVERLAP;
    }
  }
  return chunks;
}

// ── Qdrant collection setup ──────────────────────────────────────────────────

async function setupCollection(client, reset = false) {
  const collections = await client.getCollections();
  const exists = collections.collections.some((c) => c.name === COLLECTION);

  if (exists) {
    if (reset) {
      await client.deleteCollection(COLLECTION);
      console.log(`Collection '${COLLECTION}' wiped.`);
    } else {
      return;
    }
  }

  await client.createCollection(COLLECTION, {
    vectors: { size: EMBED_DIM, distance: "Cosine" },
  });
  console.log(`Collection '${COLLECTION}' created.`);
}

// ── Steps 3 & 4: embed + store ───────────────────────────────────────────────

async function ingest(pdfPaths, reset = false) {
  const client = new QdrantClient({
    url: `${QDRANT_HOST}:${QDRANT_PORT}`,
  });
  const embedder = await pipeline("feature-extraction", EMBED_MODEL, {
    dtype: "fp32",
  });

  await setupCollection(client, reset);

  for (const pdfPath of pdfPaths) {
    console.log(`\nIngesting: ${basename(pdfPath)}`);

    const pages = await extractPages(pdfPath);
    console.log(`  ${pages.length} pages with text`);

    const chunks = chunkPages(pages);
    console.log(`  ${chunks.length} chunks created`);

    const points = [];
    for (const chunk of chunks) {
      const output = await embedder(chunk.text, {
        pooling: "mean",
        normalize: true,
      });
      const vector = Array.from(output.data);

      points.push({
        id: randomUUID(),
        vector,
        payload: chunk,
      });
    }

    await client.upsert(COLLECTION, { points });
    console.log(`  ✓ ${points.length} points stored in Qdrant`);
  }

  console.log("\nIngest complete.");
}

// ── CLI entry point ──────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const reset = args.includes("--reset");
const paths = args.filter((a) => !a.startsWith("--"));

if (paths.length === 0) {
  console.log(
    "Usage:\n  node ingest.js <pdf-files...> [--reset]"
  );
  process.exit(1);
}

ingest(paths, reset);
