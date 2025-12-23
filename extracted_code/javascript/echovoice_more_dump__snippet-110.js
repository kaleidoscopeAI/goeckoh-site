export function computeTorque(engine: Engine, input: number[]) {
  const mem = engine.retrieveClosestMemory(input); // From crystals FAISS-like
  const novelty = 1 - (input.reduce((sum, v, i) => sum + v * mem[i], 0) / (len(input) * len(mem))); // Cos sim
  const arousal = engine.avgEmotional().ar;
  const tau = novelty * arousal;
  return tau > 0.5 ? tau : 0; // Gate threshold
