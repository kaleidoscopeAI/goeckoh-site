export function metaEvolve(engine: Engine) {
  const perfGrad = engine.computeH() - engine.prevH; // Delta H
  const valenceGrad = engine.avgEmotional().v; // Use valence as signal
  engine.etaSem *= (1 + 0.01 * valenceGrad * perfGrad); // Adapt rate
  engine.dt = clip(engine.dt * (1 - 0.005 * perfGrad), 0.001, 0.1); // Stabilize
