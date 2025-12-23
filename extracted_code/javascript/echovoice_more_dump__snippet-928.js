const prompt = `Generate semantic deltas for nodes: ${JSON.stringify(batch)}. Keep deltas small and relevant.`;
const response = await fetch(`${this.url}/api/generate`, {
  method: 'POST',
  body: JSON.stringify({ model: this.model, prompt })
});
const data = await response.json();
const suggestions: Record<string, number[]> = JSON.parse(data.response); // Assume parsed deltas
return suggestions;
