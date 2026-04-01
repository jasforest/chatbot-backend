# 🤖 OpenAI Chatbot – Full Setup Guide

## Files Overview

```
chatbot/
├── api/
│   └── chat.js       ← Vercel serverless function (backend)
├── widget.html       ← Chat widget (embed on your website)
├── vercel.json       ← Vercel config
└── README.md
```

---

## Step 1: Deploy to Vercel (Free)

### 1a. Install Vercel CLI
```bash
npm install -g vercel
```

### 1b. Login to Vercel
```bash
vercel login
```

### 1c. Deploy from the chatbot folder
```bash
cd chatbot
vercel
```
Follow the prompts. When asked about settings, accept the defaults.
After deployment, Vercel gives you a URL like: `https://your-project.vercel.app`

---

## Step 2: Add Your OpenAI API Key

1. Go to https://vercel.com → your project → **Settings → Environment Variables**
2. Add a new variable:
   - **Name:** `OPENAI_API_KEY`
   - **Value:** your OpenAI key (starts with `sk-...`)
3. Click **Save** and **Redeploy**

---

## Step 3: Update the Widget

Open `widget.html` and find this line near the bottom:

```js
const API_URL = "https://YOUR-PROJECT.vercel.app/api/chat";
```

Replace `YOUR-PROJECT` with your actual Vercel project name.

---

## Step 4: Embed on Your Website

### Option A — Embed as a full page
Upload `widget.html` to your site and link to it.

### Option B — Embed as a floating widget on any existing page
Copy everything from `<style>` to the closing `</script>` tag in `widget.html`
and paste it into your existing HTML page before `</body>`.

---

## Customization

**Change the AI personality** — edit this line in `api/chat.js`:
```js
content: "You are a helpful assistant on this website..."
```

**Change the model** — edit this line in `api/chat.js`:
```js
model: "gpt-4o-mini",  // cheapest
// model: "gpt-4o",    // smarter, costs more
```

**Change the bot name/avatar** — edit the header section in `widget.html`:
```html
<div class="chat-avatar">🤖</div>
<div class="chat-header-name">AI Assistant</div>
```

**Restrict CORS to your domain** — in `api/chat.js`, replace:
```js
res.setHeader("Access-Control-Allow-Origin", "*");
// with:
res.setHeader("Access-Control-Allow-Origin", "https://yoursite.com");
```

---

## Cost Estimate

Using `gpt-4o-mini`:
- Input: ~$0.15 per 1M tokens
- Output: ~$0.60 per 1M tokens
- A typical conversation message ≈ 200–500 tokens
- 1,000 messages ≈ $0.15–$0.50 total
