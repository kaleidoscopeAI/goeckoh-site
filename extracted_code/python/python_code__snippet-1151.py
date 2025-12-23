class UnifiedVoiceCrystal:
    """Complete voice adaptation and prosody system"""
    
    def __init__(self):
        self.voice_samples = {
            "neutral": [],
            "calm": [],
            "excited": []
        }
        self.prosody_profiles = {
            "neutral": {"pitch_mean": 120.0, "pitch_std": 20.0, "energy": 0.5},
            "calm": {"pitch_mean": 100.0, "pitch_std": 10.0, "energy": 0.3},
            "excited": {"pitch_mean": 180.0, "pitch_std": 40.0, "energy": 0.8}
        }
        self.adaptation_rate = 0.01
        self.voice_adaptations = deque(maxlen=100)
    
    def select_style(self, emotional_state: EmotionalState) -> str:
        """Select voice style based on emotional state with enhanced logic"""
        # Calculate composite emotional scores
        negative_arousal = emotional_state.anxiety + emotional_state.fear + emotional_state.overwhelm
        positive_arousal = emotional_state.joy + emotional_state.trust
        cognitive_load = 1.0 - emotional_state.focus
        
        # High distress → calming style
        if negative_arousal > 1.5:
            return "calm"
        
        # High positive affect → excited style
        elif positive_arousal > 1.2 and emotional_state.focus > 0.5:
            return "excited"
        
        # High cognitive load or moderate distress → neutral with calming
        elif cognitive_load > 0.7 or negative_arousal > 0.8:
            return "neutral"
        
        # Default to neutral
        else:
            return "neutral"
    
    def adapt_voice(self, audio_sample: np.ndarray, style: str, adaptation_context: str = ""):
        """Lifelong voice adaptation with enhanced learning"""
        # Validate audio sample
        if audio_sample is None or len(audio_sample) == 0:
            return
            
        # Normalize audio sample
        audio_sample = audio_sample / (np.max(np.abs(audio_sample)) + 1e-8)
        
        # Store voice samples with quality assessment
        if len(self.voice_samples[style]) < 64:  # Increased capacity
            quality_score = self._assess_audio_quality(audio_sample)
            if quality_score > 0.3:  # Minimum quality threshold
                self.voice_samples[style].append({
                    'audio': audio_sample,
                    'timestamp': time.time(),
                    'quality': quality_score,
                    'context': adaptation_context
                })
                
                # Enhanced prosody adaptation
                self._adapt_prosody_profile(audio_sample, style, quality_score)
        
        # Track adaptations with metadata
        self.voice_adaptations.append({
            'style': style,
            'timestamp': time.time(),
            'context': adaptation_context,
            'sample_count': len(self.voice_samples[style])
        })
        
        # Periodic profile optimization
        if len(self.voice_adaptations) % 10 == 0:
            self._optimize_prosody_profiles()
    
    def _assess_audio_quality(self, audio_sample: np.ndarray) -> float:
        """Assess audio quality for adaptation"""
        # Calculate various quality metrics
        energy = np.mean(audio_sample ** 2)
        zcr = np.mean(np.abs(np.diff(np.sign(audio_sample))))  # Zero crossing rate
        spectral_centroid = self._calculate_spectral_centroid(audio_sample)
        
        # Quality scoring (higher is better)
        energy_score = min(1.0, energy * 10)
        zcr_score = 1.0 - min(1.0, zcr / 0.5)  # Lower ZCR is better
        spectral_score = min(1.0, spectral_centroid / 2000)  # Reasonable spectral range
        
        return (energy_score + zcr_score + spectral_score) / 3.0
    
    def _calculate_spectral_centroid(self, audio_sample: np.ndarray) -> float:
        """Calculate spectral centroid for quality assessment"""
        fft = np.fft.fft(audio_sample)
        freqs = np.fft.fftfreq(len(audio_sample))
        magnitude = np.abs(fft)
        
        # Only consider positive frequencies
        pos_mask = freqs > 0
        if np.any(pos_mask):
            return np.sum(freqs[pos_mask] * magnitude[pos_mask]) / np.sum(magnitude[pos_mask])
        return 0.0
    
    def _adapt_prosody_profile(self, audio_sample: np.ndarray, style: str, quality_score: float):
        """Adapt prosody profiles based on new audio sample"""
        current_profile = self.prosody_profiles[style]
        adaptation_factor = self.adaptation_rate * quality_score
        
        # Extract prosodic features from audio sample
        pitch_estimate = self._estimate_pitch(audio_sample)
        energy_estimate = np.mean(audio_sample ** 2)
        
        # Adaptive learning with quality weighting
        if pitch_estimate > 0:
            pitch_error = pitch_estimate - current_profile["pitch_mean"]
            current_profile["pitch_mean"] += adaptation_factor * pitch_error
            
        energy_error = energy_estimate - current_profile["energy"]
        current_profile["energy"] += adaptation_factor * energy_error * 0.1
        
        # Update pitch variability
        pitch_std_error = np.std([self._estimate_pitch(s['audio']) for s in self.voice_samples[style] if len(s) > 0]) - current_profile["pitch_std"]
        current_profile["pitch_std"] += adaptation_factor * pitch_std_error * 0.5
        
        # Ensure reasonable bounds
        current_profile["pitch_mean"] = np.clip(current_profile["pitch_mean"], 50, 300)
        current_profile["pitch_std"] = np.clip(current_profile["pitch_std"], 5, 50)
        current_profile["energy"] = np.clip(current_profile["energy"], 0.1, 1.0)
    
    def _estimate_pitch(self, audio_sample: np.ndarray) -> float:
        """Simple pitch estimation using autocorrelation"""
        if len(audio_sample) < 100:
            return 0.0
            
        # Autocorrelation-based pitch estimation
        autocorr = np.correlate(audio_sample, audio_sample, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after zero lag
        min_period = int(16000 / 400)  # 400 Hz max
        max_period = int(16000 / 50)   # 50 Hz min
        
        if len(autocorr) > max_period:
            peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
            if peak_idx > 0:
                pitch_hz = 16000 / peak_idx
                return pitch_hz
                
        return 0.0
    
    def _optimize_prosody_profiles(self):
        """Optimize prosody profiles based on collected samples"""
        for style, samples in self.voice_samples.items():
            if len(samples) >= 5:
                # Calculate statistics from collected samples
                pitches = []
                energies = []
                
                for sample_data in samples:
                    if isinstance(sample_data, dict) and 'audio' in sample_data:
                        pitch = self._estimate_pitch(sample_data['audio'])
                        energy = np.mean(sample_data['audio'] ** 2)
                        
                        if pitch > 0:
                            pitches.append(pitch)
                        energies.append(energy)
                
                if pitches:
                    # Update profile with statistical averages
                    current_profile = self.prosody_profiles[style]
                    optimization_rate = 0.1  # Conservative optimization
                    
                    current_profile["pitch_mean"] = (
                        (1 - optimization_rate) * current_profile["pitch_mean"] +
                        optimization_rate * np.mean(pitches)
                    )
                    current_profile["pitch_std"] = (
                        (1 - optimization_rate) * current_profile["pitch_std"] +
                        optimization_rate * np.std(pitches)
                    )
                    current_profile["energy"] = (
                        (1 - optimization_rate) * current_profile["energy"] +
                        optimization_rate * np.mean(energies)
                    )
    
    def synthesize_with_prosody(self, text: str, style: str, emotional_state: EmotionalState) -> np.ndarray:
        """Synthesize speech with advanced prosody transfer"""
        profile = self.prosody_profiles[style]
        
        # Enhanced prosody modulation based on emotional state
        pitch_mod = 1.0 + 0.3 * (emotional_state.joy - emotional_state.fear)
        energy_mod = 1.0 + 0.4 * emotional_state.trust
        tempo_mod = 1.0 + 0.2 * (emotional_state.focus - 0.5)  # Focus affects tempo
        
        # Apply stress and rhythm patterns
        stress_pattern = self._generate_stress_pattern(text, emotional_state)
        
        adjusted_pitch = profile["pitch_mean"] * pitch_mod
        adjusted_energy = profile["energy"] * energy_mod
        adjusted_tempo = tempo_mod
        
        # Generate audio with enhanced synthesis
        duration = len(text) * 0.08 * adjusted_tempo  # Tempo affects duration
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Multi-component synthesis for richer sound
        t = np.linspace(0, duration, num_samples)
        
        # Fundamental frequency with vibrato
        vibrato_freq = 5.0 + 2.0 * emotional_state.joy  # Vibrato varies with joy
        vibrato_depth = 0.02 * emotional_state.trust
        pitch_variation = adjusted_pitch * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t))
        
        # Generate harmonics for richer voice
        audio = np.zeros(num_samples)
        for harmonic in range(1, 4):  # Add first 3 harmonics
            harmonic_amp = adjusted_energy / harmonic  # Higher harmonics are quieter
            harmonic_freq = pitch_variation * harmonic
            audio += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add formant structure for vowel-like quality
        formant_freqs = [800, 1200, 2400]  # F1, F2, F3 approximate
        formant_bws = [100, 150, 200]     # Formant bandwidths
        
        for i, (freq, bw) in enumerate(zip(formant_freqs, formant_bws)):
            formant_env = np.exp(-((t * freq - freq) / bw) ** 2)
            audio *= (1 + 0.3 * formant_env)  # Apply formant shaping
        
        # Apply stress pattern
        stress_envelope = self._apply_stress_pattern(num_samples, stress_pattern, sample_rate)
        audio *= stress_envelope
        
        # Add natural noise and breathing
        noise_level = 0.005 * (1 + emotional_state.anxiety)  # Anxiety increases noise
        audio += np.random.randn(num_samples) * noise_level
        
        # Apply natural envelope with attack and decay
        attack_time = 0.05  # 50ms attack
        decay_time = 0.3   # 300ms decay
        
        attack_samples = int(attack_time * sample_rate)
        decay_samples = int(decay_time * sample_rate)
        
        envelope = np.ones(num_samples)
        # Attack
        if attack_samples < num_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Decay
        if decay_samples < num_samples:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        audio *= envelope
        
        # Normalize and convert
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)
    
    def _generate_stress_pattern(self, text: str, emotional_state: EmotionalState) -> List[float]:
        """Generate stress pattern based on text and emotional state"""
        words = text.split()
        if not words:
            return [1.0]
            
        # Base stress pattern (content words get more stress)
        stress_pattern = []
        for word in words:
            # Content words (nouns, verbs, adjectives) get more stress
            if len(word) > 3:  # Simple heuristic for content words
                base_stress = 1.2
            else:
                base_stress = 0.8
            
            # Modulate by emotional state
            emotional_mod = 1.0 + 0.2 * (emotional_state.joy - emotional_state.anxiety)
            
            stress_pattern.append(base_stress * emotional_mod)
        
        return stress_pattern
    
    def _apply_stress_pattern(self, num_samples: int, stress_pattern: List[float], sample_rate: int) -> np.ndarray:
        """Apply stress pattern to audio envelope"""
        if not stress_pattern:
            return np.ones(num_samples)
            
        # Calculate samples per stress unit
        samples_per_stress = num_samples // len(stress_pattern)
        envelope = np.zeros(num_samples)
        
        for i, stress in enumerate(stress_pattern):
            start_idx = i * samples_per_stress
            end_idx = min((i + 1) * samples_per_stress, num_samples)
            
            # Smooth stress transitions
            if i > 0:
                # Smooth transition from previous stress
                prev_stress = stress_pattern[i-1]
                transition_samples = min(100, samples_per_stress // 4)
                transition = np.linspace(prev_stress, stress, transition_samples)
                envelope[start_idx:start_idx + transition_samples] = transition
                start_idx += transition_samples
            
            envelope[start_idx:end_idx] = stress
        
        return envelope

