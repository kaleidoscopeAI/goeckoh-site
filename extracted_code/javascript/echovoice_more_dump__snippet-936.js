for (const n of this.nodes.values()) {
  let localH = 0;
  for (const neighId of n.neighbors) {
    const edge = this.edges.find(e => (e.a === n.id && e.b === neighId) || (e.b === n.id && e.a === neighId))!;
    const nb = this.nodes.get(neighId)!;
    const simBit = this.bitSim(n.e, nb.e);
    const distSq = len(sub(n.pos, nb.pos)) ** 2;
    const attr = edge.w * (-edge.j * n.s * nb.s);
    localH += 0.5 * (this.lambdaBit * (1 - simBit) + this.lambdaPos * distSq + attr);
  }
  localH += this.alphaTether * len(sub(n.pos, [0, 0, 0])) ** 2;
  n.energy = localH;
}
