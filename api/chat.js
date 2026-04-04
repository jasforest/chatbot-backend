import { readFileSync } from "fs";
import path from "path";

const FALLBACK =
  "I don't have information on that.";

/** Project root on Vercel and locally — more reliable than import.meta.url for reading data files */
const INDEX_PATH = path.join(process.cwd(), "knowledge", "embeddings.json");

const EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";
const TOP_K = Number(process.env.RETRIEVAL_TOP_K || process.env.PINECONE_TOP_K || 15);
/** Chunks below this similarity are dropped (strict). Paraphrases often score ~0.2–0.35. */
const MIN_SCORE = Number(
  process.env.MIN_RETRIEVAL_SCORE ?? process.env.MIN_PINECONE_SCORE ?? 0.22
);
/**
 * If nothing passes MIN_SCORE, still use the top few chunks when the best score is at least this.
 * Stops “word for word” behavior without feeding noise when the query is totally unrelated.
 */
const WEAK_MIN = Number(process.env.WEAK_RETRIEVAL_MIN ?? 0.15);
const WEAK_TOP_K = Number(process.env.WEAK_RETRIEVAL_TOP_K ?? 5);
/** Added to cosine so word overlap (e.g. visa, immigration) can reorder chunks when embeddings tie. */
const KEYWORD_BOOST = Number(process.env.RETRIEVAL_KEYWORD_BOOST ?? 0.08);

const STOPWORDS = new Set([
  "the",
  "and",
  "for",
  "are",
  "but",
  "not",
  "you",
  "all",
  "can",
  "her",
  "was",
  "one",
  "our",
  "had",
  "how",
  "what",
  "when",
  "where",
  "which",
  "who",
  "whom",
  "this",
  "that",
  "these",
  "those",
  "with",
  "from",
  "have",
  "has",
  "had",
  "does",
  "did",
  "will",
  "into",
  "than",
  "then",
  "them",
  "they",
  "your",
  "any",
  "some",
  "such",
  "only",
  "same",
  "each",
  "both",
  "few",
  "more",
  "most",
  "other",
  "about",
  "after",
  "also",
  "being",
  "been",
  "here",
  "there",
]);

function queryKeywordsForOverlap(q) {
  return q
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));
}

/** 0..1 fraction of query keywords that appear in chunk text */
function lexicalOverlapRatio(queryText, chunkText) {
  const keys = queryKeywordsForOverlap(queryText);
  if (keys.length === 0) return 0;
  const hay = chunkText.toLowerCase();
  let hits = 0;
  for (const k of keys) {
    if (hay.includes(k)) hits++;
  }
  return hits / keys.length;
}

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

function searchLocal(queryVector, index, queryText) {
  const list = index.chunks || [];
  const scored = list.map((c) => {
    const cosine = cosineSimilarity(queryVector, c.embedding);
    const lex = lexicalOverlapRatio(queryText, c.text);
    const combined = cosine + KEYWORD_BOOST * lex;
    return {
      cosine,
      lex,
      combined,
      metadata: {
        text: c.text,
        source_file: c.metadata?.source_file || "policy",
        source_url: c.metadata?.source_url,
      },
    };
  });

  const passesStrict = (m) =>
    m.cosine >= MIN_SCORE ||
    (m.cosine >= MIN_SCORE - 0.06 && m.lex >= 0.25);

  const candidates = scored.filter(passesStrict);
  if (candidates.length > 0) {
    candidates.sort((a, b) => b.combined - a.combined);
    return candidates.slice(0, TOP_K);
  }

  scored.sort((a, b) => b.cosine - a.cosine);
  const bestCosine = scored[0]?.cosine ?? 0;
  if (bestCosine < WEAK_MIN) return [];

  return scored
    .slice(0, WEAK_TOP_K)
    .filter((m) => m.cosine >= WEAK_MIN);
}

function getLastUserMessage(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user" && typeof messages[i]?.content === "string") {
      return messages[i].content.trim();
    }
  }
  return "";
}

/** Collect recent user turns for embedding (follow-ups like "what about fees?" need prior topic). */
function getUserTurns(messages, maxTurns = 4) {
  const out = [];
  for (let i = messages.length - 1; i >= 0 && out.length < maxTurns; i--) {
    if (messages[i]?.role === "user" && typeof messages[i]?.content === "string") {
      out.unshift(messages[i].content.trim());
    }
  }
  return out;
}

const FOLLOWUP_HINT =
  /\b(that|those|this|these|it|same|also|more|above|earlier|before|you said|your (last|previous)|follow[- ]?up|and what about|how about)\b/i;

/**
 * Single string for vector search — combines recent user questions when the latest is short or refers back.
 */
function buildRetrievalQuery(messages) {
  const last = getLastUserMessage(messages);
  if (!last) return "";
  const turns = getUserTurns(messages, 4);
  if (turns.length <= 1) return last.slice(0, 8000);

  const needsContext =
    last.length < 55 ||
    FOLLOWUP_HINT.test(last) ||
    /\b(url|link|links|source|website|reference)\b/i.test(last) ||
    /^(what|how|why|when|where|who|is|are|can|do|does)\s+(about|else|that|this)\b/i.test(last.trim());

  if (!needsContext) return last.slice(0, 8000);

  const combined = turns.join("\n\n");
  return combined.slice(-8000);
}

/**
 * Small talk / meta questions — not answered from policy chunks.
 * Returns null when the message should go through RAG.
 */
function conversationalReply(text) {
  const t = text.trim();
  if (!t) return null;

  if (
    /^(hi|hello|hey|hiya|good\s+(morning|afternoon|evening))([\s!.,?]|$|\s+(there|everyone|all))*\s*$/i.test(
      t
    ) &&
    t.length < 80
  ) {
    return "Hello! I answer questions about government policy using our official policy documents. Ask me something specific from those policies when you're ready.";
  }

  if (
    /what\s+(can|do)\s+you\s+(do|help)/i.test(t) ||
    /^how\s+can\s+you\s+help/i.test(t) ||
    /^who\s+are\s+you\??$/i.test(t) ||
    /^what\s+is\s+this\??$/i.test(t) ||
    /^help\s*$/i.test(t)
  ) {
    return "I only answer using the policy information in our database—not general knowledge. Ask about a specific policy, rule, eligibility, or process. If it isn't in our documents, I'll say I don't have that information.";
  }

  if (/^(thanks?|thank\s+you|thx|ty)[\s!.]*$/i.test(t)) {
    return "You're welcome. Ask another policy question any time.";
  }

  if (/^(bye|goodbye|see\s+you)[\s!.]*$/i.test(t)) {
    return "Goodbye. Come back if you have policy questions.";
  }

  return null;
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
  const seenUrls = new Set();
  let i = 1;
  for (const m of matches) {
    const text = m.metadata?.text;
    if (typeof text !== "string" || !text.trim()) continue;
    const src = m.metadata?.source_file || "policy";
    const url = m.metadata?.source_url;
    let urlLine = "";
    if (typeof url === "string" && url.startsWith("http") && !seenUrls.has(url)) {
      seenUrls.add(url);
      urlLine = `\nOfficial source URL (for this document): ${url}`;
    }
    parts.push(`[${i}] (source file: ${src})${urlLine}\n${text.trim()}`);
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

  const smallTalk = conversationalReply(userQuestion);
  if (smallTalk) {
    return res.status(200).json({ reply: smallTalk });
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
    const retrievalQuery = buildRetrievalQuery(messages);
    const queryVector = await openaiEmbedding(retrievalQuery || userQuestion);
    const matches = searchLocal(queryVector, index, retrievalQuery || userQuestion);

    if (matches.length === 0) {
      return res.status(200).json({ reply: FALLBACK });
    }

    const context = buildContext(matches);
    if (!context) {
      return res.status(200).json({ reply: FALLBACK });
    }

    const systemPrompt = `You are a government policy assistant. Your answers must be based ONLY on the policy excerpts in "Context" below.

Rules:
- Use the conversation history only to understand follow-up questions (e.g. what "that" or "it" refers to). Do not treat earlier assistant messages as a source of policy facts—only the Context below may be cited for policies.
- Use only information that is explicitly stated in the Context. Do not use outside knowledge.
- Do not suggest, recommend, infer, or add information that is not directly supported by the Context.
- If the Context does not contain information that reasonably relates to the question, reply with exactly this sentence and nothing else: ${FALLBACK}
- If the Context states relevant requirements, steps, or criteria (even if not a perfect checklist), summarize only what is stated there.
- Keep answers concise and factual. If you cite details, they must appear in the Context.
- Do not mention "Context" or chunk numbers unless the user explicitly asks how the system works.
- When "Official source URL" appears above a chunk, you may repeat that exact URL when the user asks for a link or source. Do not invent URLs that are not shown in the Context.
- If the user asks whether a policy is still current or may have changed, only repeat what the Context says; do not speculate.

Context:
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
