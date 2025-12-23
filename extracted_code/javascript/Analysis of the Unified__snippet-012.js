const upstream = await fetch(url, {
  method: req.method,
  headers: { 'Content-Type': 'application/json', ...(req.headers.authorization ? { Authorization: req.headers.authorization } : {}) },
  body: (req.method === 'GET' || req.method === 'HEAD') ? undefined : JSON.stringify(req.body)
});
const text = await upstream.text();
res.status(upstream.status).send(text);
