  for (const n of this.nodes.values()) {
    n.vel = n.vel.map(v => v * (1 - this.gammaDecoh * this.dt) + gaussian() * 0.001); // Dephase
  }
