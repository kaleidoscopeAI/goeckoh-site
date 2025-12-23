  const avgC = // as before
  this.engineAlpha = this.engineAlpha.map((a, i) => a * (1 + this.emotionalGamma * Math.tanh(avgC[i % this.emotionalSpecies])));
