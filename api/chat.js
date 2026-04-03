import { readFileSync } from "fs";
import path from "path";

const FALLBACK =
  "I don't have information on that.";

/** Project root on Vercel and locally — more reliable than import.meta.url for reading data files */
const INDEX_PATH = path.join(process.cwd(), "knowledge", "embeddings.json");

const EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";
const TOP_K = Number(process.env.RETRIEVAL_TOP_K || process.env.PINECONE_TOP_K || 6);
const MIN_SCORE = Number(
  process.env.MIN_RETRIEVAL_SCORE ?? process.env.MIN_PINECONE_SCORE ?? 0.28
);

/** @type {{ embeddingModel?: string; chunks?: Array<{ id: string; text: string; embedding: number[]; metadata: { source_file: string; chunk_index: number } }> } | null} */
let cachedIndex = null;

function loadIndex() {
  if (cachedIndex) return cachedIndex;
  const raw = readFileSync(INDEX_PATH, "utf8");
  cachedIndex = JSON.parse(raw);
  return cachedIndex;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function searchLocal(queryVector, index) {
  const list = index.chunks || [];
  const scored = list.map((c) => ({
    score: cosineSimilarity(queryVector, c.embedding),
    metadata: {
      text: c.text,
      source_file: c.metadata?.source_file || "policy",
    },
  }));
  scored.sort((x, y) => y.score - x.score);
  return scored.slice(0, TOP_K).filter((m) => m.score >= MIN_SCORE);
}

function getLastUserMessage(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user" && typeof messages[i]?.content === "string") {
      return messages[i].content.trim();
    }
  }
  return "";
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

function buildContext(matches) {
  const parts = [];
  let i = 1;
  for (const m of matches) {
    const text = m.metadata?.text;
    if (typeof text !== "string" || !text.trim()) continue;
    const src = m.metadata?.source_file || "policy";
    parts.push(`[${i}] (source: ${src})\n${text.trim()}`);
    i++;
  }
  return parts.join("\n\n---\n\n");
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Invalid request body" });
  }

  if (!process.env.OPENAI_API_KEY) {
    return res.status(500).json({ error: "Server misconfiguration" });
  }

  const userQuestion = getLastUserMessage(messages);
  if (!userQuestion) {
    return res.status(400).json({ error: "No user message found" });
  }

  let index;
  try {
    index = loadIndex();
  } catch {
    return res.status(503).json({
      error:
        "Knowledge index missing. Run `npm run ingest` locally and deploy `knowledge/embeddings.json`.",
    });
  }

  if (!index.chunks?.length) {
    return res.status(503).json({ error: "Knowledge index is empty. Run `npm run ingest`." });
  }

  if (index.embeddingModel && index.embeddingModel !== EMBEDDING_MODEL) {
    console.warn(
      `Index was built with ${index.embeddingModel} but OPENAI_EMBEDDING_MODEL is ${EMBEDDING_MODEL}`
    );
  }

  try {
    const queryVector = await openaiEmbedding(userQuestion);
    const matches = searchLocal(queryVector, index);

    if (matches.length === 0) {
      return res.status(200).json({ reply: FALLBACK });
    }

    const context = buildContext(matches);
    if (!context) {
      return res.status(200).json({ reply: FALLBACK });
    }

    const systemPrompt = `You are a government policy assistant. Your answers must be based ONLY on the policy excerpts in "Context" below.

Rules:
- Use only information that is explicitly stated in the Context. Do not use outside knowledge.
- Do not suggest, recommend, infer, or add information that is not directly supported by the Context.

- If the Context does not contain enough information to answer the question, reply with exactly this sentence and nothing else: ${FALLBACK}
- Keep answers concise and factual. If you cite details, they must appear in the Context.
- Do not mention "Context" or chunk numbers unless the user explicitly asks how the system works.
- When answering, always end with: "Source: [url]" using the URL provided in the context metadata.
- If the user sends a greeting (e.g. "hello", "hi"), respond politely and ask how you can help with government policy questions. Do not apply the fallback message for greetings.
- If the user asks about something unrelated to government policy, politely let them know you can only assist with government policy questions.
- If the user asks about current status of a policy, only state what is written in the Context. Do not speculate about whether information is still current or has changed.
${context}`;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: CHAT_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          ...messages,
        ],
        max_tokens: 500,
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res.status(response.status).json({ error: err.error?.message || "OpenAI error" });
    }

    const data = await response.json();
    const reply = data.choices[0].message.content?.trim() || FALLBACK;

    res.status(200).json({ reply });
  } catch (error) {
    console.error("API error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
}
