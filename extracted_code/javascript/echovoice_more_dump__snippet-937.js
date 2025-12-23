const grads1 = this.computePosGrads();
for (const n of this.nodes.values()) {
  const g1 = grads1.get(n.id)!;
  const acc1 = scale(g1, -1 / n.mass); // -grad for descent
  n.vel = add(n.vel, scale(acc1, this.dt / 2));
  n.pos = add(n.pos, scale(n.vel, this.dt));
}
const grads2 = this.computePosGrads();
for (const n of this.nodes.values()) {
  const g2 = grads2.get(n.id)!;
  const acc2 = scale(g2, -1 / n.mass);
  n.vel = add(n.vel, scale(acc2, this.dt / 2));
  n.vel = scale(n.vel, this.damping);
  n.vel = add(n.vel, [gaussian() * 0.01, gaussian() * 0.01, gaussian() * 0.01]); // Noise
}
