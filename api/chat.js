export default async function handler(req, res) {
  // Allow cross-origin requests from your website
  res.setHeader("Access-Control-Allow-Origin", "*"); // Replace * with your domain in production e.g. "https://yoursite.com"
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Invalid request body" });
  }

  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`, // Set this in Vercel dashboard
      },
      body: JSON.stringify({
        model: "gpt-4o-mini", // Cheapest model — swap to gpt-4o if needed
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant on this website. Be concise, friendly, and helpful.", // Customize this!
          },
          ...messages,
        ],
        max_tokens: 500,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      return res.status(response.status).json({ error: err.error?.message || "OpenAI error" });
    }

    const data = await response.json();
    const reply = data.choices[0].message.content;
    res.status(200).json({ reply });
  } catch (error) {
    console.error("API error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
}
