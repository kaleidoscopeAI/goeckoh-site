for (const n of this.nodes.values()) {
  const f = (yc: number[]) => {
    const rhs = new Array(this.emotionalSpecies).fill(0);
    for (let sp = 0; sp < this.emotionalSpecies; sp++) {
      let transportIn = 0, transportOut = 0;
      for (const neighId of n.neighbors) {
        const nb = this.nodes.get(neighId)!;
        const tIj = 0.05; // Fixed transport rate
        transportIn += tIj * nb.c[sp];
        transportOut += tIj * yc[sp];
      }
      const sExt = gaussian() * 0.01; // Stimuli
      rhs[sp] = this.emotionalP - this.emotionalD * yc[sp] + transportIn - transportOut + sExt;
    }
    return rhs;
  };
  n.c = rk4Step(f, n.c, this.dt);
}
// Modulate lambdas
const avgC = Array.from(this.nodes.values()).reduce((sum, n) => sum.map((v, i) => v + n.c[i]), new Array(this.emotionalSpecies).fill(0)).map(v => v / this.nodes.size);
this.lambdaBit *= (1 + this.emotionalGamma * Math.tanh(avgC[0])); // Example f(C)
this.lambdaPos *= (1 + this.emotionalGamma * Math.tanh(avgC[0])); // g(C)
