if (this.analyzedHypotheses.size === 0) return 0;

let totalConfidence = 0;
let count = 0;

for (const [_, analysis] of this.analyzedHypotheses) {
  if (Date.now() - analysis.timestamp < 30000) {
    totalConfidence += analysis.confidence;
    count++;
  }
}

return count > 0 ? totalConfidence / count : 0;
