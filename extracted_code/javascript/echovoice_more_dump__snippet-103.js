  for (const n of this.nodes.values()) {
    n.symbols = ['concept' + Math.floor(n.energy)]; // Auto-extract placeholder
  }
