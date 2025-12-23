const forces = new Map<string, Vec3>();
for (const [id, n] of this.nodes) forces.set(id, [0, -0.01, 0]); // Gravity
for (const e of this.edges) {
  const na = this.nodes.get(e.a), nb = this.nodes.get(e.b);
  if (!na || !nb) continue;
  const r = sub(na.pos, nb.pos);
  const dist = len(r) + 1e-9;
  const u = normalize(r);
  const mag = -e.k * (dist - e.L);
  const fij = scale(u, mag);
  forces.set(na.id, add(forces.get(na.id)!, fij));
  forces.set(nb.id, sub(forces.get(nb.id)!, fij));
}
return forces;
