for (const n of this.nodes.values()) {
  let acc = [0,0,0];
  for (const other of this.nodes.values()) {
    if (n.id === other.id) continue;
    const r = sub(n.pos, other.pos);
    const dist = len(r) + 1e-9;
    acc = add(acc, scale(normalize(r), -1 / dist**2));  // Gravity-like for clusters
  }
  n.vel = add(n.vel, scale(acc, this.dt));
}
