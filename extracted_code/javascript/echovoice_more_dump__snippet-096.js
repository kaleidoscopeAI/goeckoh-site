  const perf = this.computeH(); // Low H good
  this.dt *= (1 + 0.01 * (this.thetaE - perf)); // Evolve dt via valence grad
