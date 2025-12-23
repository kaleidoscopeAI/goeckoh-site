// src/engines/EmbeddedQuantumEngine.ts
import { EmbeddedOllamaEngine } from '../embedded/ollamaEngine';

export class EmbeddedQuantumEngine {
  private nodeCount: number;
  private quantumStates: Array<{ phase: number; coherence: number }>;
  private ollamaEngine: EmbeddedOllamaEngine;
  private knowledgeCrystals: number = 0;
  private globalCoherence: number = 0.5;
  private hypothesisHistory: any[] = [];
  private analyzedHypotheses: Map<string, any> = new Map();
  private dynamicParameters = {
    emotionalValenceBoost: 0,
    quantumEntanglementStrength: 1.0,
    mimicryForceModifier: 1.0,
    coherenceDecayRate: 0.99
  };

  constructor(nodeCount: number) {
    this.nodeCount = nodeCount;
    this.quantumStates = Array.from({ length: nodeCount }, () => ({
      phase: Math.random() * Math.PI * 2,
      coherence: 0.5
    }));
    
    this.ollamaEngine = new EmbeddedOllamaEngine();
    this.initializeOllama();
  }

  private async initializeOllama(): Promise<void> {
    try {
      await this.ollamaEngine.start();
      await this.ollamaEngine.ensureModel('llama2');
      console.log('✅ Embedded Ollama engine ready');
    } catch (error) {
      console.warn('❌ Embedded Ollama failed, using cognitive fallback:', error);
    }
  }

  async update(positions: Float32Array, emotionalField: any): Promise<void> {
    const sampleSize = Math.min(1000, this.nodeCount);
    let totalCoherence = 0;

    for (let i = 0; i < sampleSize; i++) {
      const idx = Math.floor((i * this.nodeCount) / sampleSize);
      const state = this.quantumStates[idx];
      
      // Enhanced emotional modulation
      const emotionalBoost = this.dynamicParameters.emotionalValenceBoost;
      state.phase += (emotionalField.valence * 0.1 + emotionalField.arousal * 0.05) * 
                    (1 + emotionalBoost);
      
      // Update coherence with dynamic decay
      state.coherence *= this.dynamicParameters.coherenceDecayRate;
      
      // Calculate local coherence from spatial relationships
      const localCoherence = this.calculateLocalCoherence(idx, positions);
      state.coherence = 0.9 * state.coherence + 0.1 * localCoherence;
      
      totalCoherence += state.coherence;
      
      // Knowledge crystallization with embedded Ollama
      if (this.shouldCrystallizeKnowledge(idx, state)) {
        await this.crystallizeKnowledge(idx, emotionalField);
      }
    }
    
    this.globalCoherence = totalCoherence / sampleSize;
  }

  private calculateLocalCoherence(nodeIdx: number, positions: Float32Array): number {
    let localCoherence = 0;
    let neighborCount = 0;
    const i3 = nodeIdx * 3;
    
    // Sample nearby nodes for coherence calculation
    for (let j = 0; j < Math.min(20, this.nodeCount); j += 5) {
      const otherIdx = (nodeIdx + j) % this.nodeCount;
      if (otherIdx === nodeIdx) continue;
      
      const j3 = otherIdx * 3;
      const dx = positions[i3] - positions[j3];
      const dy = positions[i3 + 1] - positions[j3 + 1];
      const dz = positions[i3 + 2] - positions[j3 + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (dist < 80) {
        const otherState = this.quantumStates[otherIdx];
        const phaseDiff = Math.abs(this.quantumStates[nodeIdx].phase - otherState.phase);
        const entanglement = this.dynamicParameters.quantumEntanglementStrength;
        localCoherence += Math.cos(phaseDiff) * entanglement;
        neighborCount++;
      }
    }
    
    return neighborCount > 0 ? localCoherence / neighborCount : 0.5;
  }

  private shouldCrystallizeKnowledge(nodeIdx: number, state: any): boolean {
    const baseProbability = 0.001 * state.coherence;
    const hypothesisBoost = this.getHypothesisConfidenceBoost();
    return Math.random() < (baseProbability * (1 + hypothesisBoost));
  }

  private getHypothesisConfidenceBoost(): number {
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
  }

  private async crystallizeKnowledge(triggerNode: number, emotionalField: any): Promise<void> {
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
  }

  private integrateOllamaFeedback(analysis: any): void {
    const adjustments = analysis.analysis?.parameter_adjustments || {};
    
    // Smooth integration of parameter adjustments
    this.dynamicParameters.emotionalValenceBoost = 
      0.8 * this.dynamicParameters.emotionalValenceBoost + 
      0.2 * (adjustments.emotional_valence_boost || 0);
      
    this.dynamicParameters.quantumEntanglementStrength = 
      0.9 * this.dynamicParameters.quantumEntanglementStrength + 
      0.1 * (1 + (adjustments.quantum_entanglement_strength || 0));
      
    this.dynamicParameters.mimicryForceModifier = 
      0.9 * this.dynamicParameters.mimicryForceModifier + 
      0.1 * (1 + (adjustments.mimicry_force_modifier || 0));
  }

  private generateHypothesis(triggerNode: number): any {
    const hypotheses = [
      `Quantum coherence threshold crossed at node ${triggerNode}`,
      `Emergent pattern formation in cognitive sector ${triggerNode % 8}`,
      `Emotional resonance creating coherence spike near node ${triggerNode}`,
      `Knowledge structure crystallization imminent in region ${triggerNode}`,
      `Phase synchronization suggesting global awareness emergence`
    ];
    
    const hypothesis = {
      text: hypotheses[Math.floor(Math.random() * hypotheses.length)],
      timestamp: Date.now(),
      confidence: this.globalCoherence,
      triggerNode
    };
    
    this.hypothesisHistory.push(hypothesis);
    if (this.hypothesisHistory.length > 50) {
      this.hypothesisHistory.shift();
    }
    
    return hypothesis;
  }

  generateHypotheses(count: number = 5): any[] {
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
  }

  getQuantumInfluence(particleA: number, particleB: number): number {
    const stateA = this.quantumStates[particleA];
    const stateB = this.quantumStates[particleB];
    const phaseCorrelation = Math.cos(stateA.phase - stateB.phase);
    return phaseCorrelation * stateA.coherence * stateB.coherence;
  }

  getGlobalCoherence(): number { return this.globalCoherence; }
  getKnowledgeCrystals(): number { return this.knowledgeCrystals; }
  getDynamicParameters(): any { return { ...this.dynamicParameters }; }
  getOllamaStatus(): any { return this.ollamaEngine.getStatus(); }

  async shutdown(): Promise<void> {
    await this.ollamaEngine.stop();
  }
}
