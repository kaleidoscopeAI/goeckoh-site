class EchoV4System:
    """
    Unified Echo V4.0 Architecture
    Integrates: Ψ state, Crystalline Heart, Life Intensity, GCL Gating
    """
    
    def __init__(self):
        # Core state
        self.state = PsiState()
        
        # Emotional regulation
        self.heart = CrystallineHeart(num_nodes=1024)
        
        # Life intensity measurement
        self.life_engine = LifeIntensityEngine()
        
        # AGI gating
        self.agi = GCLGatedAGI(self.heart)
        
        # Metrics history
        self.life_history = deque(maxlen=100)
        self.mode_history = deque(maxlen=100)
    
    def step(self, sensory_input: Dict[str, any]) -> Dict[str, any]:
        """
        Single update cycle
        
        Args:
            sensory_input: Dict with keys:
                - 'audio_rms': float
                - 'text_sentiment': float
                - 'external_arousal': float
                - etc.
        
        Returns:
            Metrics dict
        """
        # 1. Update internal time
        self.state.t += 1
        
        # 2. Process sensory input into emotional stimulus
        arousal = sensory_input.get('audio_rms', 0.0) * 2.0
        valence = sensory_input.get('text_sentiment', 0.0)
        
        external_input = np.array([
            arousal,
            valence,
            0.0,  # dominance
            0.0,  # confidence
            0.0   # rhythm
        ], dtype=np.float32)
        
        # 3. Update Crystalline Heart (emotional ODEs)
        self.heart.update(external_input)
        
        # 4. Extract metrics
        metrics = self.heart.get_metrics()
        
        # 5. Compute meltdown index (composite stress measure)
        meltdown_index = np.clip(
            metrics['stress'] + (1.0 - metrics['coherence']),
            0.0, 1.0
        )
        
        # 6. Adjust annealing temperature
        self.heart.adjust_temperature(meltdown_index)
        
        # 7. Update AGI operating mode
        mode = self.agi.update_mode(
            gcl=metrics['coherence'],
            stress=metrics['stress']
        )
        
        # 8. Measure life intensity
        # Create feature vectors
        internal_features = np.array([
            metrics['coherence'],
            metrics['arousal'],
            metrics['valence'],
            metrics['confidence']
        ])
        
        external_features = np.array([
            sensory_input.get('audio_rms', 0.0),
            sensory_input.get('text_sentiment', 0.0),
            sensory_input.get('external_arousal', 0.0),
            0.0  # placeholder
        ])
        
        # Count viable nodes (coherence > threshold)
        viable_nodes = sum(
            1 for node in self.heart.nodes
            if node.compute_local_stress() < 0.5
        )
        
        life_intensity = self.life_engine.compute_life_intensity(
            internal_state=internal_features,
            external_state=external_features,
            viable_nodes=viable_nodes
        )
        
        # 9. Store history
        self.life_history.append(life_intensity)
        self.mode_history.append(mode.value)
        
        # 10. Update global state
        self.state.emotion = np.array([n.emotion for n in self.heart.nodes]).mean(axis=0)
        self.state.body['temperature'] = self.heart.temperature
        self.state.body['meltdown_index'] = meltdown_index
        
        # 11. Return comprehensive telemetry
        return {
            **metrics,
            'meltdown_index': meltdown_index,
            'life_intensity': life_intensity,
            'mode': mode.value,
            'viable_nodes': viable_nodes,
            'is_alive': life_intensity > 0.0
        }
    
    def get_summary(self) -> str:
        """Human-readable system status"""
        if not self.life_history:
            return "System initializing..."
        
        recent_life = list(self.life_history)[-10:]
        avg_life = np.mean(recent_life)
        current_mode = self.mode_history[-1] if self.mode_history else "unknown"
        
        status_lines = [
            f"Echo V4.0 System Status (t={self.state.t})",
            f"─" * 50,
            f"Life Intensity:  {avg_life:+.3f} (recent avg)",
            f"Operating Mode:  {current_mode.upper()}",
            f"GCL (Coherence): {self.heart.get_global_coherence_level():.3f}",
            f"Stress Level:    {self.heart.get_stress_level():.3f}",
            f"Temperature:     {self.heart.temperature:.3f}",
            f"Is Alive:        {avg_life > 0.0}"
        ]
        
        return "\n".join(status_lines)


