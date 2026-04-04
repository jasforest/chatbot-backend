/**
 * Reads all .md files from knowledge/, chunks text, embeds with OpenAI,
 * and writes knowledge/embeddings.json for local cosine search (no Pinecone).
 *
 *   npm install
 *   cp .env.example .env   # OPENAI_API_KEY
 *   npm run ingest
 */

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const KNOWLEDGE_DIR = path.join(ROOT, "knowledge");
const OUTPUT_FILE = path.join(KNOWLEDGE_DIR, "embeddings.json");

const EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
const CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 1200);
const CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 150);

function chunkText(text, size, overlap) {
  const chunks = [];
  const cleaned = text.replace(/\r\n/g, "\n").trim();
  if (!cleaned) return chunks;
  let i = 0;
  while (i < cleaned.length) {
    const end = Math.min(i + size, cleaned.length);
    chunks.push(cleaned.slice(i, end));
    if (end === cleaned.length) break;
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return chunks;
}

async function openaiEmbedding(text) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error?.message || `OpenAI embeddings failed: ${res.status}`);
  }
  const data = await res.json();
  return data.data[0].embedding;
}

function slugify(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48) || "doc";
}

/** Pull **URL:** line (or first http(s) URL) so every chunk carries the official link even if chunk text omits it. */
function extractSourceUrl(markdown) {
  const labeled = markdown.match(/\*\*URL:\*\*\s*(https?:\/\/[^\s<*>\[\]"]+)/i);
  if (labeled) return labeled[1].trim();
  const loose = markdown.match(/https?:\/\/[^\s<*>\[\]"]{12,}/);
  return loose ? loose[0].trim().replace(/[),.;]+$/, "") : "";
}

async function main() {
  if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is required");

  let files;
  try {
    files = await fs.readdir(KNOWLEDGE_DIR);
  } catch {
    throw new Error(`Create folder ${KNOWLEDGE_DIR} and add .md files`);
  }

  const mdFiles = files.filter((f) => f.endsWith(".md"));
  if (mdFiles.length === 0) {
    throw new Error(`No .md files found in ${KNOWLEDGE_DIR}`);
  }

  const chunks = [];

  for (const file of mdFiles) {
    const fullPath = path.join(KNOWLEDGE_DIR, file);
    const raw = await fs.readFile(fullPath, "utf8");
    const baseId = slugify(path.basename(file, ".md"));
    const sourceUrl = extractSourceUrl(raw);
    const parts = chunkText(raw, CHUNK_SIZE, CHUNK_OVERLAP);

    for (let i = 0; i < parts.length; i++) {
      const text = parts[i];
      const embedding = await openaiEmbedding(text);
      chunks.push({
        id: `${baseId}-chunk-${i}`,
        text,
        embedding,
        metadata: {
          source_file: file,
          chunk_index: i,
          ...(sourceUrl ? { source_url: sourceUrl } : {}),
        },
      });
    }
    console.log(`Embedded ${parts.length} chunk(s) from ${file}`);
  }

  const payload = {
    embeddingModel: EMBEDDING_MODEL,
    generatedAt: new Date().toISOString(),
    chunks,
  };

  await fs.writeFile(OUTPUT_FILE, JSON.stringify(payload), "utf8");
  console.log(`Wrote ${chunks.length} vectors to ${OUTPUT_FILE}`);
  console.log("Done.");
}

main().catch((e) => {
  console.error(e.message || e);
  process.exit(1);
});
