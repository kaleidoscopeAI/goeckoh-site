let h = 0;
for (const edge of this.edges) {
  const na = this.nodes.get(edge.a)!, nb = this.nodes.get(edge.b)!;
  const simBit = this.bitSim(na.e, nb.e);
  const distSq = len(sub(na.pos, nb.pos)) ** 2;
  const attr = edge.w * (-edge.j * na.s * nb.s); // Spin-glass optional
  h += this.lambdaBit * (1 - simBit) + this.lambdaPos * distSq + attr;
}
for (const n of this.nodes.values()) {
  const tether = this.alphaTether * len(sub(n.pos, [0, 0, 0])) ** 2; // x_i0 = 0 for simplicity
  h += tether; // + reg terms if needed
}
return h;
