  for (const n of this.nodes.values()) {
    n.v = clip(n.v + gaussian() * 0.1 * n.ar, -1, 1); // Valence drift with arousal
    n.ar = clip(n.ar - 0.05 + 0.1 * Math.abs(n.energy), 0, 1); // Arousal from E
    n.st = clip(n.st + 0.05 * (n.k - 0.5), -1, 1); // Stance from confidence
    n.coh = 1 - Math.abs(n.v - n.st); // Coherence as alignment
  }
  // Embodied: mock CPU load
  const cpuLoad = Math.random(); // Real: from os.loadavg()
  this.lambdaBit *= (1 + 0.01 * cpuLoad); // Feedback hardware to params
