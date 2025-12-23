  const oldW = edge.w;
  const na = this.nodes.get(edge.a)!, nb = this.nodes.get(edge.b)!;
  const bitTerm = this.lambdaBit * (1 - this.bitSim(na.e, nb.e));
  const posTerm = this.lambdaPos * len(sub(na.pos, nb.pos)) ** 2;
  const oldAttr = oldW * (-edge.j * na.s * nb.s);
  const newAttr = newW * (-edge.j * na.s * nb.s);
  return bitTerm + posTerm + newAttr - (bitTerm + posTerm + oldAttr); // Delta = new - old for that edge
