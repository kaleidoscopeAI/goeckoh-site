const recent = this.hypothesisHistory.slice(-count);

return recent.map(hyp => {
  const analysis = this.analyzedHypotheses.get(hyp.text);
  return analysis ? {
    ...hyp,
    refined: analysis.analysis.refined_hypothesis,
    plausibility: analysis.analysis.plausibility,
    analyzed: true
  } : {
    ...hyp,
    analyzed: false
  };
});
