// dev/proxy-ollama.js
import express from 'express';
import fetch from 'node-fetch';

const app = express();
const PORT = 5174;
const OLLAMA = 'http://localhost:11434';

app.use(express.json({ limit: '30mb' }));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

app.all('/api/*', async (req, res) => {
  const url = `${OLLAMA}${req.path.replace(/^\/api/, '')}`;
  try {
    const upstream = await fetch(url, {
      method: req.method,
      headers: { 'Content-Type': 'application/json', ...(req.headers.authorization ? { Authorization: req.headers.authorization } : {}) },
      body: (req.method === 'GET' || req.method === 'HEAD') ? undefined : JSON.stringify(req.body)
    });
    const text = await upstream.text();
    res.status(upstream.status).send(text);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/health', (_, res) => res.json({ status: 'ok', ts: new Date().toISOString() }));

app.listen(PORT, () => {
  console.log(`Ollama proxy running -> http://localhost:${PORT} -> ${OLLAMA}`);
});
