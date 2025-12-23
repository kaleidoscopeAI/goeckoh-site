  const stateStr = JSON.stringify(batch);
  const torque = computeTorque(this.engine, inputVec);  // From torque.ts
  const prompt = `Reflect on state: ${stateStr}. Torque ${torque} indicates dissonance. Suggest deltas for self-modeling.`;
  const res = await fetch(`${this.url}/api/generate`, { method: 'POST', body: JSON.stringify({ model: this.model, prompt }) });
  const data = await res.json();
  const deltas = JSON.parse(data.response);  // Parse deltas
  // Re-embed for meta: mock vec from text
  return deltas;
