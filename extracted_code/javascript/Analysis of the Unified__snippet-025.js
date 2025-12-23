try {
  const prompt = this.buildPrompt(hypothesis, systemContext);
  const body = { model: 'llama2', prompt, stream: false, options: { temperature: 0.7, top_p: 0.9, num_predict: 150 } };
  const res = await rateLimitedFetch('http://localhost:5174/api/generate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  }, { retries: 2, backoff: 300 });
  const text = await res.text();
  const analysis = safeExtractJson(text) || this.localFallback(hypothesis, systemContext);
  const payload = { original: hypothesis, analysis, timestamp: Date.now(), confidence: (analysis.plausibility || 0.5) * (hypothesis.confidence || 0.5) };
  this.worker.postMessage({ cmd: 'ollama_feedback', requestId, analysis: payload });
} catch (err) {
  console.warn('Mediator failed:', err);
  const fallback = this.localFallback(hypothesis, systemContext);
  this.worker.postMessage({ cmd: 'ollama_feedback', requestId, analysis: fallback });
}
