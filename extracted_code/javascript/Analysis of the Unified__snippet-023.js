const adjustments = analysis.analysis?.parameter_adjustments || {};
this.dynamicParameters.emotionalValenceBoost = 0.8 * this.dynamicParameters.emotionalValenceBoost + 0.2 * (adjustments.emotional_valence_boost || 0);
this.dynamicParameters.quantumEntanglementStrength = 0.9 * this.dynamicParameters.quantumEntanglementStrength + 0.1 * (1 + (adjustments.quantum_entanglement_strength || 0));
this.dynamicParameters.mimicryForceModifier = 0.9 * this.dynamicParameters.mimicryForceModifier + 0.1 * (1 + (adjustments.mimicry_force_modifier || 0));
// store analysis
if (analysis?.original?.text) this.analyzedHypotheses.set(analysis.original.text, analysis);
