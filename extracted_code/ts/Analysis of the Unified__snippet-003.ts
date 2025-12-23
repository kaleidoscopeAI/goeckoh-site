import { rateLimitedFetch } from './ollamaClient';
import { safeExtractJson } from '../util/safeJson';

export class OllamaMediator {
  worker: Worker;
  constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = this.handleWorkerMessage.bind(this);
  }
  handleWorkerMessage(e: MessageEvent) {
    const msg = e.data;
    if (!msg?.cmd) return;
    if (msg.cmd === 'hypothesis') {
      this.handleHypothesis(msg.hypothesis, msg.systemContext, msg.requestId);
    } else if (msg.cmd === 'log') console.log('[worker]', msg.msg);
  }
  async handleHypothesis(hypothesis: any, systemContext: any, requestId: number) {
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
  }
  buildPrompt(hypothesis: any, context: any) {
    return `You are a cognitive scientist. SYSTEM: GC=${context.globalCoherence}, V=${context.emotionalField.valence}, A=${context.emotionalField.arousal}, KC=${context.knowledgeCrystals}. HYPOTHESIS: "${hypothesis.text}". Respond with JSON {...}`;
  }
  localFallback(hypothesis: any, context: any) {
    const plausibility = 0.5 + Math.random() * 0.3;
    return { original: hypothesis, analysis: { plausibility, coherence_impact: plausibility > 0.6 ? 'positive' : 'neutral', refined_hypothesis: `[Local] ${hypothesis.text}`, parameter_adjustments: { emotional_valence_boost: (plausibility - 0.5) * 0.2, quantum_entanglement_strength: 0, mimicry_force_modifier: 0 }, reasoning: 'local fallback' }, timestamp: Date.now(), confidence: plausibility * (hypothesis.confidence || 0.5) };
  }
}
