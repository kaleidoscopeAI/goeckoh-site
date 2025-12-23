try {
  const jsonMatch = response.match(/\{[\s\S]*\}/);
  if (!jsonMatch) throw new Error('No JSON found in response');

  const analysis = JSON.parse(jsonMatch[0]);

  return {
    original: { text: originalHypothesis, confidence: context.globalCoherence },
    analysis,
    timestamp: Date.now(),
    confidence: (analysis.plausibility || 0.5) * context.globalCoherence
  };
} catch (error) {
  console.warn('Failed to parse Ollama response, using fallback:', error);
  return this.createFallbackAnalysis(originalHypothesis, context);
}
