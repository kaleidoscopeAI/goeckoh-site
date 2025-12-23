return {  // Real for p5 viz: particles with colors from E
  particles: Array.from(this.nodes.values()).map(n => ({
    pos: n.pos, color: `hsl(${math.map(n.energy, 0, 3, 120, 0)}, 100%, 50%)`,  // Green-red
    size: n.a * 10, halo: n.k > 0.5  // Bool for halo
  })),
  // ... Edges, stats
};
