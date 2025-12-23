for (const [idx, n] of [...this.nodes.values()].entries()) {
  const hist = this.history[idx];
  hist.push({ pos: n.pos, grad: len(this.computePosGrads().get(n.id)!) });
  if (hist.length > this.crystalWindow) hist.shift();
  if (hist.length === this.crystalWindow) {
    const meanPos = hist[0].pos.map((_, i) => hist.reduce((sum, h) => sum + h.pos[i], 0) / this.crystalWindow);
    const varX = hist.reduce((sum, h) => sum + len(sub(h.pos, meanPos)) ** 2, 0) / this.crystalWindow;
    const meanG = hist.reduce((sum, h) => sum + h.grad, 0) / this.crystalWindow;
    const bitEnt = this.bitProbabilistic ? -n.e.reduce((sum, p) => sum + p * Math.log(p + 1e-10) + (1 - p) * Math.log(1 - p + 1e-10), 0) / this.dBit : 0;
    if (varX < this.crystalEpsX && meanG < this.crystalEpsG && bitEnt < this.crystalEpsB) {
      this.alphaTether *= 10; // Freeze by strong tether
      // Persist if needed
    }
  }
}
