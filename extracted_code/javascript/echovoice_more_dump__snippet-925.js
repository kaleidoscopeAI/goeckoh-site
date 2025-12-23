for (const n of this.nodes.values()) {
  let E = 0;
  for (const e of this.edges) {
    if (e.a === n.id || e.b === n.id) {
      const other = this.nodes.get(e.a === n.id ? e.b : e.a)!;
      const r = len(sub(n.pos, other.pos));
      const dl = r - e.L;
      E += 0.5 * e.k * dl * dl;
    }
  }
  n.energy = E;
  n.K *= this.knowledge_decay;
}
const raw = Array.from(this.nodes.values()).map(n => 6 * n.energy + 2 * n.K + gaussian() * 0.02);
const maxv = Math.max(...raw);
const exps = raw.map(v => Math.exp(v - maxv));
const sum = exps.reduce((a, b) => a + b, 0) || 1;
let idx = 0;
for (const n of this.nodes.values()) {
  n.A = exps[idx++] / sum;
  const ell = this.replicate_lambda * ((n.K - this.theta_K) + (n.A - this.theta_A) - (n.energy - this.theta_E));
  n.repProb = 1 / (1 + Math.exp(-ell));
  if (n.repProb > Math.random()) this.replicate(n);
}
