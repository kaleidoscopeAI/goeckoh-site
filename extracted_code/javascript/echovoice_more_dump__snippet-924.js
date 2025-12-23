const F = this.computeForces();
for (const [id, n] of this.nodes) {
  const Fi = F.get(id)!;
  const acc = scale(Fi, 1 / n.mass);
  n.pos = add(n.pos, add(scale(n.vel, this.dt), scale(acc, 0.5 * this.dt * this.dt)));
}
const F2 = this.computeForces();
for (const [id, n] of this.nodes) {
  const a1 = scale(F.get(id)!, 1 / n.mass);
  const a2 = scale(F2.get(id)!, 1 / n.mass);
  n.vel = scale(add(n.vel, scale(add(a1, a2), 0.5 * this.dt)), this.damping);
}
