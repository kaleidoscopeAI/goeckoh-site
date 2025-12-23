this.computeLocalEnergies();
const globalH = this.computeH();
// Knowledge K: avg confidence (for prob) or low entropy
for (const n of this.nodes.values()) {
  if (this.bitProbabilistic) {
    n.k = n.e.reduce((sum, p) => sum + p * (1 - p), 0) / this.dBit; // Entropy approx, low=high K
    n.k = 1 - n.k; // Invert to confidence
  } else n.k *= this.knowledgeDecay; // Decay as before
  // Awareness A: softmax over energy + engines
  n.a = Math.exp(-n.energy + n.b + n.h + n.kappa + n.mu);
}
const sumA = Array.from(this.nodes.values()).reduce((sum, n) => sum + n.a, 0) || 1;
for (const n of this.nodes.values()) {
  n.a /= sumA;
  const ell = this.replicateLambda * ((n.k - this.thetaK) + (n.a - this.thetaA) - (n.energy / globalH - this.thetaE));
  n.repProb = 1 / (1 + Math.exp(-ell));
  n.pruneRisk = n.energy / globalH > 0.1 ? 0.5 : 0; // High energy risk
}
