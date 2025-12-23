  const highT = this.temperature * 2;
  const highSnap = {...this.snapshot()}; // Simulate high T step
  const deltaH = this.computeH() - highSnap.stats.totalH;
  const pSwap = Math.exp(-deltaH * (1/this.temperature - 1/highT));
  if (Math.random() < pSwap) this.loadSnapshot(highSnap); // Exchange
