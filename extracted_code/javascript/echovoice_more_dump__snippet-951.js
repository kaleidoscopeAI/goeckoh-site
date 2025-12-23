this.t += this.dt;
this.emotionalStep(); // 1. Emotions RK4
// 2. Modulate coeffs done in emotional
this.stepPhysics(); // 3. Positions leapfrog
this.enginesStep(); // 5. Engines RK4
this.bitStep(); // 6. Bits Metropolis
// 7. Quantum skip
this.semanticStep(); // PDF diffusion
this.computeMetrics(); // A, K, repProb, etc.
for (const n of [...this.nodes.values()]) this.replicate(n);
if (Math.floor(this.t / this.dt) % this.birthPeriod === 0) {
  // Propose new node
  const parent = [...this.nodes.values()][Math.floor(Math.random() * this.nodes.size)];
  const newPos = add(parent.pos, [gaussian(), gaussian(), gaussian()]);
  const newE = parent.e.map(v => v + gaussian() * 0.1);
  const tempN = this.addNode(newPos, newE);
  const deltaH = this.computeH() - this.computeH(); // Approx
  if (Math.random() > Math.min(1, Math.exp(-deltaH / this.temperature * 2))) this.removeNode(tempN.id); // Higher T_birth
}
if (Math.floor(this.t / this.dt) % this.sleepPeriod === 0) this.consolidation();
this.temperature *= this.annealRate;
