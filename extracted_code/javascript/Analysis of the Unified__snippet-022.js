const recent = this.hypothesisHistory.slice(-count);
return recent.map(h => {
  const analysis = this.analyzedHypotheses.get(h.text);
  return analysis ? { ...h, refined: analysis.analysis.refined_hypothesis, plausibility: analysis.analysis.plausibility, analyzed: true } : { ...h, analyzed: false };
});
