const hypothesis = this.generateHypothesis(triggerNode);
this.knowledgeCrystals++;

// Use embedded Ollama for real-time analysis
try {
  const systemContext = {
    globalCoherence: this.globalCoherence,
    emotionalField,
    knowledgeCrystals: this.knowledgeCrystals
  };

  const analysis = await this.ollamaEngine.analyzeCognitiveHypothesis(hypothesis.text, systemContext);
  this.analyzedHypotheses.set(hypothesis.text, analysis);
  this.integrateOllamaFeedback(analysis);

  console.log('🧠 Embedded Analysis:', analysis.analysis.refined_hypothesis);

} catch (error) {
  console.warn('Embedded analysis failed:', error);
  // Continue with local processing
}
