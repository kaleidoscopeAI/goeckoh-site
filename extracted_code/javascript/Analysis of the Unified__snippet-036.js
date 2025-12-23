if (!this.isRunning) {
  throw new Error('Ollama engine not running');
}

const response = await fetch(`http://127.0.0.1:11435${endpoint}`, {
  method,
  headers: { 'Content-Type': 'application/json' },
  body: data ? JSON.stringify(data) : undefined
});

if (!response.ok) {
  throw new Error(`Ollama API error: ${response.statusText}`);
}

return response.json();
