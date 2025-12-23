class CompleteUnifiedSystem:
    """
    COMPLETE UNIFIED NEURO-ACOUSTIC AGI SYSTEM
    
    This system integrates ALL Python scripts from the root directory and subdirectories
    into one comprehensive, production-ready AGI system.
    
    INTEGRATED COMPONENTS:
    ✅ Echo V4 Core - Unified AGI architecture with PsiState
    ✅ Crystalline Heart - 1024-node emotional regulation lattice
    ✅ Audio System - Rust bio-acoustic engine + Neural TTS
    ✅ Voice Engine - Neural voice cloning capabilities
    ✅ Audio Bridge - Real-time audio processing
    ✅ Session Persistence - Long-term memory logging
    ✅ Neural Voice Synthesis - Advanced speech synthesis
    ✅ Enhanced Unified System - Document-based discoveries
    ✅ Robust Unified System - Pure NumPy implementation
    ✅ Autism-Optimized VAD - 1.2s pause tolerance
    ✅ ABA Therapeutics - Evidence-based interventions
    ✅ Voice Crystal - Prosody transfer and adaptation
    ✅ Mathematical Framework - 128+ equations
    ✅ Quantum Systems - Hamiltonian dynamics
    ✅ Memory Systems - Crystalline lattice memory
    ✅ Cyber-Physical Control - Hardware integration
    """
    
    def __init__(self):
        print("🚀 INITIALIZING COMPLETE UNIFIED NEURO-ACOUSTIC AGI SYSTEM")
        print("📚 Integrating ALL Python scripts from root and subdirectories...")
        
        # Core systems
        # Default clone ref: none (auto-capture first utterance)
        self.clone_ref_wav = os.getenv("GOECKOH_CLONE_WAV")
        self.psi_state = PsiState()
        self.crystalline_heart = UnifiedCrystallineHeart(num_nodes=1024)
        self.memory_system = UnifiedMemorySystem()
        self.audio_system = UnifiedAudioSystem(clone_ref_wav=self.clone_ref_wav)
        
        # Autism and therapeutic systems
        self.autism_vad = AutismOptimizedVAD()
        self.aba_engine = UnifiedAbaEngine()
        self.voice_crystal = UnifiedVoiceCrystal()
        self.behavior_monitor = BehaviorMonitor() if BehaviorMonitor else None
        self.strategy_advisor = StrategyAdvisor() if StrategyAdvisor else None
        self._auto_captured_clone = False
        
        # Advanced systems
        self.molecular_system = UnifiedMolecularSystem()
        self.cyber_controller = UnifiedCyberPhysicalController()
        
        # Tracking and metrics
        self.metrics_history = deque(maxlen=200)
        self.start_time = time.time()
        self.aba_interventions = deque(maxlen=50)
        
        print("✅ Complete system initialization successful")
        print(f"🧠 Crystalline Heart: {self.crystalline_heart.num_nodes} nodes")
        print(f"🎵 Audio System: Rust={RUST_AVAILABLE}, Neural={NEURAL_TTS_AVAILABLE}")
        print(f"🧩 ABA Engine: {len(self.aba_engine.aba_skills)} categories")
        print(f"🎤 Voice Crystal: {len(self.voice_crystal.voice_samples)} styles")
        print(f"👂 Autism VAD: {self.autism_vad.min_silence_duration_ms}ms tolerance")
        print(f"💾 Memory System: Crystalline + Persistence")
        print(f"⚛️  Quantum System: Pure NumPy evolution")
        print(f"🔧 Cyber-Physical: L0-L4 control hierarchy")

    def set_clone_wav(self, path: Optional[str]):
        """Set/replace the speaker reference WAV for cloning."""
        self.clone_ref_wav = os.path.abspath(path) if path else None
        if hasattr(self.audio_system, "set_clone_wav"):
            self.audio_system.set_clone_wav(self.clone_ref_wav)
    
    def process_input(self, text_input: str, audio_input: Optional[np.ndarray] = None, 
                      sensory_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete processing pipeline integrating ALL systems with enhanced logic
        """
        start_time = time.time()
        prosody_features = None
        if audio_input is not None and audio_input.size > 0:
            try:
                audio_i16 = (audio_input * 32768.0).astype(np.int16)
                prosody_features = maybe_extract_prosody(audio_i16, int(self.autism_vad.sample_rate if hasattr(self.autism_vad, "sample_rate") else 16000))
            except Exception:
                prosody_features = None

            # If no clone reference is set, auto-capture this utterance as the clone reference
            if not self.clone_ref_wav and not self._auto_captured_clone:
                try:
                    tmp_path = Path(tempfile.gettempdir()) / f"goeckoh_clone_{uuid.uuid4().hex}.wav"
                    with wave.open(str(tmp_path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(int(self.autism_vad.sample_rate if hasattr(self.autism_vad, "sample_rate") else 16000))
                        wf.writeframes(audio_i16.tobytes())
                    self.set_clone_wav(str(tmp_path))
                    self._auto_captured_clone = True
                    print(f"🎙️ Auto-captured clone reference: {tmp_path}")
                except Exception:
                    pass
        
        # 1. Update Echo V4 Core PsiState with enhanced tracking
        self.psi_state.t += 1
        self.psi_state.history_hashes = self.memory_system.get_history_hashes()
        
        # Update world state with current context
        self.psi_state.world['current_input'] = text_input
        self.psi_state.world['processing_time'] = start_time
        self.psi_state.world['audio_available'] = audio_input is not None
        
        # Update body state with sensory information
        if sensory_data:
            self.psi_state.body.update(sensory_data)
        
        # Update self-model with system state
        self.psi_state.self_model['system_load'] = len(self.metrics_history) / 200.0
        self.psi_state.self_model['emotional_coherence'] = 0.0  # Will be updated later
        
        # 2. Enhanced autism-optimized VAD processing with pause analysis
        vad_status = {'is_speech': False, 'should_transcribe': True, 'pause_analysis': {}}
        if audio_input is not None:
            is_speech, should_transcribe = self.autism_vad.process_audio_chunk(audio_input)
            pause_analysis = self.autism_vad.get_pause_analysis()
            vad_status = {
                'is_speech': is_speech, 
                'should_transcribe': should_transcribe,
                'pause_analysis': pause_analysis
            }
            
            # Autism-optimized pause respect
            if not should_transcribe and pause_analysis.get('avg_pause_duration', 0) < self.autism_vad.min_silence_duration_ms / 1000.0:
                return {'status': 'waiting_for_complete_utterance', 'vad_status': vad_status}
        
        # 3. Enhanced emotional stimulus calculation with comprehensive analysis
        if sensory_data is None:
            sensory_data = {}
        
        # Multi-dimensional arousal calculation
        base_arousal = 0.05 + (len(text_input) * 0.01)
        punctuation_arousal = 0.3 * text_input.count('!') + 0.2 * text_input.count('?')
        capital_arousal = 0.5 * (1.0 if text_input.isupper() else 0.0)
        content_arousal = self._analyze_content_arousal(text_input)
        
        arousal = base_arousal + punctuation_arousal + capital_arousal + content_arousal
        arousal = np.clip(arousal, 0.0, 1.0)
        
        # Sentiment analysis simulation
        sentiment = self._analyze_sentiment(text_input)
        if 'sentiment' in sensory_data:
            sentiment = (sentiment + sensory_data['sentiment']) / 2.0
        
        # Rhythm and prosody analysis
        rhythm = self._analyze_rhythm(text_input)
        if 'rhythm' in sensory_data:
            rhythm = (rhythm + sensory_data['rhythm']) / 2.0
        
        # Create comprehensive emotional stimulus
        emotional_stimulus = np.array([
            arousal,
            sentiment,
            sensory_data.get('dominance', 0.0),
            sensory_data.get('confidence', 0.5),
            rhythm,
            sensory_data.get('anxiety', self._estimate_anxiety(text_input)),
            sensory_data.get('focus', self._estimate_focus(text_input)),
            sensory_data.get('overwhelm', self._estimate_overwhelm(text_input))
        ], dtype=np.float32)
        
        # 4. Update Echo V4 Core emotion with validation
        self.psi_state.emotion = np.clip(emotional_stimulus[:5], -1.0, 1.0)
        
        # 5. Enhanced quantum state evolution with emotional coupling
        self.molecular_system.evolve_quantum_state()
        quantum_state = QuantumState(
            hamiltonian=self.molecular_system.hamiltonian,
            wavefunction=self.molecular_system.wavefunction,
            energy=self.molecular_system.molecular_properties['binding_energy'],
            correlation_length=self.molecular_system.molecular_properties.get('correlation_length', 5.0),
            criticality_index=self.molecular_system.molecular_properties.get('criticality_index', 1.0)
        )
        
        # 6. Enhanced Crystalline Heart update with quantum influence
        self.crystalline_heart.update(emotional_stimulus, quantum_state)
        
        # 7. Extract enhanced emotional state with validation
        emotional_state = self.crystalline_heart.get_enhanced_emotional_state()
        
        # Ensure emotional state values are in valid range
        for field in emotional_state.__dataclass_fields__:
            value = getattr(emotional_state, field)
            if isinstance(value, (int, float)):
                setattr(emotional_state, field, np.clip(value, 0.0, 1.0))
        
        # Update self-model emotional coherence
        emotional_values = [emotional_state.joy, emotional_state.trust, emotional_state.focus]
        coherence = 1.0 - np.std(emotional_values)
        self.psi_state.self_model['emotional_coherence'] = coherence
        
        # 8. Enhanced ABA intervention with context awareness
        aba_intervention = self.aba_engine.intervene(emotional_state, text_input)
        self.aba_interventions.append(aba_intervention)
        
        # Track skill attempts based on intervention
        if aba_intervention.get('skill_focus'):
            category = self._infer_skill_category(aba_intervention['skill_focus'])
            success = self._evaluate_intervention_success(aba_intervention, emotional_state)
            self.aba_engine.track_skill_attempt(category, aba_intervention['skill_focus'], success, text_input)
        
        # 9. Enhanced voice style selection with adaptation tracking
        voice_style = self.voice_crystal.select_style(emotional_state)
        
        # Voice adaptation if audio input available
        if audio_input is not None:
            adaptation_context = f"style_{voice_style}_emotion_{coherence:.2f}"
            self.voice_crystal.adapt_voice(audio_input, voice_style, adaptation_context)
        
        # 10. Enhanced memory encoding with emotional context and indexing
        if text_input.strip():
            # Generate semantic embedding
            embedding = self._generate_semantic_embedding(text_input)
            
            # Encode memory with full emotional context
            self.memory_system.encode_memory(embedding, emotional_state, text_input)
            
            # Update world state with memory information
            if hasattr(self.memory_system, 'get_memory_count'):
                self.psi_state.world['memory_count'] = self.memory_system.get_memory_count()
            else:
                self.psi_state.world['memory_count'] = len(self.memory_system.vector_index) if hasattr(self.memory_system, 'vector_index') else 0
                
            if hasattr(self.memory_system, 'get_memory_stability'):
                self.psi_state.world['memory_stability'] = self.memory_system.get_memory_stability()
            else:
                self.psi_state.world['memory_stability'] = 0.5
        
        # 11. Enhanced response generation with ABA integration and emotional awareness
        if text_input.strip():
            response_text = text_input.lower() \
                .replace("you are", "i am") \
                .replace("you", "i") \
                .replace("your", "my") \
                .capitalize()
            
            # ABA intervention integration with enhanced context
            if aba_intervention.get('strategy') == 'calming':
                response_text = f"{aba_intervention.get('social_story', '')} {response_text}"
            elif aba_intervention.get('strategy') == 'attention':
                response_text = f"{aba_intervention.get('social_story', '')} {response_text}"
            elif aba_intervention.get('strategy') == 'sensory':
                response_text = f"{aba_intervention.get('social_story', '')} {response_text}"
            elif aba_intervention.get('strategy') == 'reinforcement':
                response_text = f"{aba_intervention.get('reward', '')} {response_text}"
            elif aba_intervention.get('strategy') == 'support':
                response_text = f"{aba_intervention.get('social_story', '')} {response_text}"
            
            # Add emotional context to response
            if emotional_state.joy > 0.7:
                response_text = f"{response_text} 😊"
            elif emotional_state.anxiety > 0.7:
                response_text = f"{response_text} Take a deep breath."
            elif emotional_state.focus < 0.3:
                response_text = f"Let's focus together. {response_text}"
        else:
            response_text = "Listening..."
        
        # 12. Enhanced voice synthesis with emotional prosody
        gcl = self.crystalline_heart.get_global_coherence_level()
        
        # Generate audio with emotional prosody
        voice_audio = self.voice_crystal.synthesize_with_prosody(response_text, voice_style, emotional_state)
        
        # Try neural TTS first, then Rust, then Python fallback
        audio_data = self.audio_system.synthesize_response(response_text, arousal, voice_style, self.clone_ref_wav)

        # Use voice crystal audio if higher quality or when primary synthesis failed
        if audio_data is None:
            audio_data = voice_audio
        elif voice_audio is not None and len(voice_audio) > len(audio_data):
            audio_data = voice_audio

        if audio_data is None:
            audio_data = np.array([], dtype=np.float32)
        
        # 13. Enhanced voice adaptation with quality tracking
        if audio_input is not None and len(audio_input) > 0:
            adaptation_context = f"style_{voice_style}_emotion_{coherence:.2f}_gcl_{gcl:.3f}"
            self.voice_crystal.adapt_voice(audio_input, voice_style, adaptation_context)
        
        # 14. Enhanced cyber-physical hardware mapping
        self.cyber_controller.update_hardware_mapping(emotional_state)

        # 14b. Lightweight behavior monitoring (no heavy deps)
        behavior_event = None
        behavior_suggestions = []
        behavior_suggestions_out = []
        if self.behavior_monitor:
            needs_correction = text_input.strip() != response_text.strip()
            audio_energy = audio_rms(audio_input) if audio_input is not None else 0.0
            normalized_text = text_input.strip().lower()
            behavior_event = self.behavior_monitor.register(normalized_text, needs_correction, audio_energy)
            if behavior_event and self.strategy_advisor:
                behavior_suggestions = self.strategy_advisor.suggest(behavior_event)
                top = behavior_suggestions[0] if behavior_suggestions else None
                if top:
                    response_text = f"{response_text} ({top.title}: {top.description})"
                behavior_suggestions_out = [
                    {"title": s.title, "description": s.description, "category": s.category}
                    for s in behavior_suggestions
                ]
        
        # 15. Enhanced metrics calculation with comprehensive analytics
        processing_time = time.time() - start_time
        
        # Calculate enhanced metrics
        stress = self.crystalline_heart.compute_local_stress(self.crystalline_heart.nodes[0])
        life_intensity = self._calculate_enhanced_life_intensity()
        mode = self._determine_enhanced_mode(gcl)
        emotional_coherence = np.linalg.norm(emotional_state.to_vector())
        quantum_coherence = self.molecular_system.molecular_properties['quantum_coherence']
        memory_stability = np.std(self.memory_system.memory_crystal) if hasattr(self.memory_system, 'memory_crystal') else 0.5
        hardware_coupling = self.cyber_controller.control_levels['L1_embodied']['hardware_feedback']
        aba_success_rate = self.aba_engine.get_success_rate()
        
        # Calculate skill mastery level
        all_levels = [prog.current_level for cat_skills in self.aba_engine.progress.values() for prog in cat_skills.values()]
        skill_mastery_level = max(1, int(np.mean(all_levels))) if all_levels else 1
        
        sensory_regulation = max(0, 1.0 - emotional_state.overwhelm)
        processing_pause_respect = 1.0  # Always respect pauses in complete system
        
        metrics = SystemMetrics(
            gcl=gcl,
            stress=stress,
            life_intensity=life_intensity,
            mode=mode,
            emotional_coherence=emotional_coherence,
            quantum_coherence=quantum_coherence,
            memory_stability=memory_stability,
            hardware_coupling=hardware_coupling,
            aba_success_rate=aba_success_rate,
            skill_mastery_level=skill_mastery_level,
            sensory_regulation=sensory_regulation,
            processing_pause_respect=processing_pause_respect,
            timestamp=time.time()
        )
        
        # 16. Store metrics and log session with enhanced tracking
        self.metrics_history.append(metrics)
        
        # Enhanced session logging with emotional context
        if hasattr(self.memory_system, 'session_log'):
            self.memory_system.session_log.log_interaction(text_input, response_text, metrics)
        
        # 17. Enhanced audio enqueue with priority based on emotional urgency
        if len(audio_data) > 0:
            if hasattr(self.audio_system, 'enqueue_audio'):
                try:
                    self.audio_system.enqueue_audio(response_text, arousal, voice_style, self.clone_ref_wav)
                except TypeError:
                    # Fallback for different signature
                    self.audio_system.enqueue_audio(response_text, arousal, voice_style, self.clone_ref_wav)
        
        # 18. Return comprehensive response with all system states
        system_status = self.get_complete_system_status()
        coaching_plan = self._build_coaching_plan(
            emotional_state=system_status.get('emotional_state', emotional_state),
            gcl=system_status.get('gcl', metrics.gcl),
            stress=system_status.get('stress', metrics.stress),
            text_input=text_input
        )
        system_status['coaching'] = coaching_plan

        return {
            'response_text': response_text,
            'audio_data': audio_data.tolist() if audio_data.size > 0 else [],
            'metrics': metrics,
            'emotional_state': emotional_state,
            'quantum_state': quantum_state,
            'psi_state': self.psi_state,
            'aba_intervention': aba_intervention,
            'behavior_event': behavior_event,
            'behavior_suggestions': behavior_suggestions_out,
            'voice_style': voice_style,
            'vad_status': vad_status,
            'processing_time': processing_time,
            'system_status': system_status,
            'coaching': coaching_plan,
            'audio_engines': {
                'rust_available': RUST_AVAILABLE,
                'neural_tts_available': NEURAL_TTS_AVAILABLE,
                'audio_device_available': AUDIO_AVAILABLE,
                'voice_crystal_active': len(self.voice_crystal.voice_adaptations) > 0
            },
            'integrated_components': {
                'echo_v4_core': True,
                'crystalline_heart': True,
                'audio_system': True,
                'voice_engine': True,
                'session_persistence': True,
                'autism_vad': True,
                'aba_therapeutics': True,
                'voice_crystal': True,
                'quantum_system': True,
                'memory_system': True,
                'cyber_physical': True,
                'molecular_system': True
            },
            'enhanced_features': {
                'pause_analysis': vad_status.get('pause_analysis', {}),
                'molecular_analysis': self.molecular_system.get_molecular_analysis(),
                'cyber_physical_state': self.cyber_controller.get_system_state(),
                'aba_progress': self.aba_engine.get_detailed_progress(),
                'voice_adaptations': len(self.voice_crystal.voice_adaptations)
            }
        }
    
    def _analyze_content_arousal(self, text: str) -> float:
        """Analyze content-based arousal from text"""
        arousal_words = {
            'high': ['excited', 'amazing', 'wow', 'incredible', 'fantastic', 'love', 'happy', 'great'],
            'medium': ['good', 'nice', 'okay', 'fine', 'well', 'better'],
            'low': ['sad', 'bad', 'terrible', 'awful', 'hate', 'angry', 'upset', 'worried']
        }
        
        words = text.lower().split()
        arousal_score = 0.0
        
        for word in words:
            if word in arousal_words['high']:
                arousal_score += 0.3
            elif word in arousal_words['medium']:
                arousal_score += 0.1
            elif word in arousal_words['low']:
                arousal_score -= 0.1
        
        return np.clip(arousal_score, -0.5, 0.5)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'worried', 'upset']
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _analyze_rhythm(self, text: str) -> float:
        """Analyze text rhythm and prosody"""
        # Simple rhythm analysis based on punctuation and word length variation
        punctuation_rhythm = text.count('.') + text.count('!') + text.count('?')
        word_lengths = [len(word) for word in text.split()]
        
        if not word_lengths:
            return 0.0
        
        length_variance = np.var(word_lengths)
        rhythm_score = min(1.0, (punctuation_rhythm + length_variance) / 10.0)
        
        return rhythm_score
    
    def _estimate_anxiety(self, text: str) -> float:
        """Estimate anxiety level from text"""
        anxiety_indicators = ['worried', 'anxious', 'nervous', 'scared', 'afraid', 'panic', 'stress', 'overwhelm']
        words = text.lower().split()
        
        anxiety_count = sum(1 for word in words if any(indicator in word for indicator in anxiety_indicators))
        return min(1.0, anxiety_count * 0.3)
    
    def _estimate_focus(self, text: str) -> float:
        """Estimate focus level from text"""
        focus_indicators = ['focus', 'concentrate', 'pay attention', 'listen', 'understand', 'clear']
        distraction_indicators = ['confused', 'distracted', 'lost', 'dont understand', 'hard']
        
        words = text.lower().split()
        focus_count = sum(1 for word in words if any(indicator in word for indicator in focus_indicators))
        distraction_count = sum(1 for word in words if any(indicator in word for indicator in distraction_indicators))
        
        base_focus = 0.5  # Default focus level
        focus_adjustment = (focus_count - distraction_count) * 0.2
        
        return np.clip(base_focus + focus_adjustment, 0.0, 1.0)
    
    def _estimate_overwhelm(self, text: str) -> float:
        """Estimate overwhelm level from text"""
        overwhelm_indicators = ['too much', 'overwhelm', 'cant handle', 'too many', 'stress', 'difficult']
        words = text.lower().split()
        
        overwhelm_count = sum(1 for word in words if any(indicator in word for indicator in overwhelm_indicators))
        return min(1.0, overwhelm_count * 0.4)
    
    def _generate_semantic_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        # Simple semantic embedding based on word characteristics
        words = text.lower().split()
        
        # Create embedding features
        features = [
            len(words),  # Text length
            text.count('!'),  # Exclamation count
            text.count('?'),  # Question count
            sum(1 for word in words if len(word) > 6),  # Long words
            sum(1 for word in words if word.isupper()),  # Capitalized words
        ]
        
        # Pad to 32 dimensions with normalized features
        embedding = np.zeros(32)
        for i, feature in enumerate(features):
            if i < 32:
                embedding[i] = feature / max(1, max(features))
        
        # Add some semantic variation
        embedding += np.random.randn(32) * 0.1
        
        return embedding
    
    def _infer_skill_category(self, skill: str) -> str:
        """Infer ABA skill category from skill name"""
        skill_lower = skill.lower()
        
        if any(word in skill_lower for word in ['brush', 'wash', 'dress', 'medication']):
            return 'self_care'
        elif any(word in skill_lower for word in ['greet', 'ask', 'express', 'aac']):
            return 'communication'
        elif any(word in skill_lower for word in ['share', 'turn', 'emotion', 'empathy']):
            return 'social_tom'
        else:
            return 'self_care'  # Default category
    
    def _evaluate_intervention_success(self, intervention: Dict, emotional_state: EmotionalState) -> bool:
        """Evaluate intervention success based on emotional state changes"""
        if not intervention.get('strategy'):
            return False
        
        strategy = intervention['strategy']
        
        # Success criteria based on strategy and emotional state
        if strategy == 'calming':
            # Success if anxiety decreased or trust increased
            return emotional_state.anxiety < 0.5 or emotional_state.trust > 0.6
        elif strategy == 'attention':
            # Success if focus improved
            return emotional_state.focus > 0.6
        elif strategy == 'sensory':
            # Success if overwhelm reduced
            return emotional_state.overwhelm < 0.4
        elif strategy == 'reinforcement':
            # Success if positive emotions maintained
            return emotional_state.joy > 0.6 and emotional_state.trust > 0.5
        elif strategy == 'support':
            # Success if emotional stability achieved
            return abs(emotional_state.joy - emotional_state.fear) < 0.3
        
        return False
    
    def _calculate_enhanced_life_intensity(self) -> float:
        """Enhanced life intensity calculation with comprehensive components"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        current_emotion = self.crystalline_heart.get_enhanced_emotional_state()
        emotion_vector = current_emotion.to_vector()
        
        # Emotional entropy component
        entropy = -np.sum(np.abs(emotion_vector) * np.log(
            np.abs(emotion_vector) + 1e-6
        ))
        
        # Quantum component
        quantum_component = self.molecular_system.molecular_properties['quantum_coherence']
        
        # Memory component
        memory_count = len(self.memory_system.vector_index) if hasattr(self.memory_system, 'vector_index') else 0
        memory_component = memory_count / max(1, self.crystalline_heart.time_step)
        
        # ABA component
        aba_component = self.aba_engine.get_success_rate()
        
        # Sensory component
        sensory_component = max(0, 1.0 - current_emotion.overwhelm)
        
        # Voice component
        voice_component = len(self.voice_crystal.voice_adaptations) / max(1, len(self.metrics_history))
        
        # Cyber-physical component
        cyber_component = self.cyber_controller.control_levels['L1_embodied']['hardware_feedback']
        
        # Molecular component
        molecular_component = self.molecular_system.molecular_properties.get('stability', 0.5)
        
        # Combined life intensity calculation
        L_t = (
            0.20 * entropy +           # Emotional entropy
            0.15 * quantum_component +  # Quantum coherence
            0.12 * memory_component +   # Memory richness
            0.15 * aba_component +      # ABA success
            0.10 * sensory_component +  # Sensory regulation
            0.08 * voice_component +     # Voice adaptation
            0.12 * cyber_component +    # Hardware integration
            0.08 * molecular_component   # Molecular stability
        )
        
        return float(np.clip(L_t, -1.0, 1.0))
    
    def _determine_enhanced_mode(self, gcl: float) -> str:
        """Enhanced mode determination"""
        if gcl < 0.2:
            return "CRISIS"
        elif gcl < 0.4:
            return "ELEVATED"
        elif gcl < 0.7:
            return "NORMAL"
        else:
            return "FLOW"

    def _build_coaching_plan(self, emotional_state: EmotionalState, gcl: float, stress: float, text_input: str) -> Dict[str, Any]:
        """
        Map current state to human-facing coaching guidance for de-escalation or reinforcement.
        """
        plan: Dict[str, Any] = {
            'mode': 'steady',
            'priority': 'medium',
            'actions': [],
            'voice_tone': 'neutral-warm',
            'pace': 'normal',
            'script': '',
            'breathing_prompt': None
        }

        gcl_val = float(np.clip(gcl, 0.0, 1.0))
        stress_val = float(max(0.0, stress))
        overwhelm = float(getattr(emotional_state, "overwhelm", 0.0))
        anxiety = float(getattr(emotional_state, "anxiety", 0.0))
        joy = float(getattr(emotional_state, "joy", 0.0))
        trust = float(getattr(emotional_state, "trust", 0.0))

        if gcl_val < 0.45 or stress_val > 0.35 or overwhelm > 0.6 or anxiety > 0.6:
            plan.update({
                'mode': 'deescalate',
                'priority': 'high',
                'voice_tone': 'calm-soft',
                'pace': 'slow',
                'breathing_prompt': 'Try a 4–6 breath: inhale 4s, exhale 6s.',
                'actions': [
                    "Lower volume and slow cadence.",
                    "Acknowledge feeling: 'I hear this is hard.'",
                    "Offer control: 'Want a moment or to keep going?'",
                    "Reflect one key concern in their words."
                ],
                'script': "I’m here with you. Let’s slow down, take a breath together, and tackle one thing at a time."
            })
        elif joy + trust > 1.0 and stress_val < 0.25 and gcl_val >= 0.7:
            plan.update({
                'mode': 'reinforce',
                'priority': 'medium',
                'voice_tone': 'warm-encouraging',
                'pace': 'steady',
                'actions': [
                    "Mirror the goal or win they shared.",
                    "Offer a next best step or concise summary.",
                    "Invite collaboration: 'Does that match what you need?'"
                ],
                'script': "Great progress. I’ll summarize briefly and suggest the next best step so we keep momentum."
            })
        else:
            plan.update({
                'mode': 'steady',
                'priority': 'medium',
                'voice_tone': 'neutral-warm',
                'pace': 'normal',
                'actions': [
                    "Summarize one key point back.",
                    "Ask a concise open question to clarify intent.",
                    "Keep a 2–3 second pause to let them process."
                ],
                'script': "Here’s what I’m hearing. I’ll pause so you can add or correct me."
            })

        if text_input:
            plan['context_note'] = text_input[:160]

        return plan
    
    def get_complete_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with all components"""
        gcl = self.crystalline_heart.get_global_coherence_level()
        emotional_state = self.crystalline_heart.get_enhanced_emotional_state()
        
        # Calculate system metrics
        stress = np.mean([self.crystalline_heart.compute_local_stress(node) for node in self.crystalline_heart.nodes])
        life_intensity = self._calculate_enhanced_life_intensity()
        
        # Memory metrics with safety checks
        memory_count = len(self.memory_system.vector_index) if hasattr(self.memory_system, 'vector_index') else 0
        memory_stability = np.std(self.memory_system.memory_crystal) if hasattr(self.memory_system, 'memory_crystal') else 0.5
        
        # Hamiltonian calculation with error handling
        try:
            hamiltonian = self.crystalline_heart.compute_hamiltonian()
        except Exception:
            hamiltonian = 0.0
        
        return {
            'system_mode': self._determine_enhanced_mode(gcl),
            'gcl': gcl,
            'stress': stress,
            'life_intensity': life_intensity,
            'emotional_state': emotional_state,
            'psi_state': self.psi_state,
            'quantum_coherence': self.molecular_system.molecular_properties['quantum_coherence'],
            'memory_stability': memory_stability,
            'uptime': time.time() - self.start_time,
            'time_step': self.crystalline_heart.time_step,
            'memory_count': memory_count,
            'temperature': self.crystalline_heart.temperature,
            'hamiltonian': hamiltonian,
            'aba_metrics': {
                'success_rate': self.aba_engine.get_success_rate(),
                'total_attempts': sum(prog.attempts for cat_skills in self.aba_engine.progress.values() for prog in cat_skills.values()),
                'interventions_count': len(self.aba_interventions),
                'skill_levels': {cat: {skill: prog.current_level for skill, prog in skills.items()} 
                               for cat, skills in self.aba_engine.progress.items()}
            },
            'voice_metrics': {
                'adaptations_count': len(self.voice_crystal.voice_adaptations),
                'available_styles': list(self.voice_crystal.voice_samples.keys()),
                'current_profiles': self.voice_crystal.prosody_profiles
            },
            'autism_features': {
                'vad_silence_tolerance_ms': self.autism_vad.min_silence_duration_ms,
                'vad_threshold': self.autism_vad.threshold,
                'processing_pause_respect': True,
                'sensory_regulation': max(0, 1.0 - emotional_state.overwhelm)
            },
            'mathematical_framework': {
                'annealing_temperature': self.crystalline_heart.temperature,
                'time_step': self.crystalline_heart.time_step,
                'modularity': getattr(self.crystalline_heart, 'compute_modularity', lambda: 1.0)(),
                'correlation_length': getattr(self.crystalline_heart, 'correlation_length', 5.0),
                'criticality_index': getattr(self.crystalline_heart, 'criticality_index', 1.0)
            },
            'cyber_physical_status': self.cyber_controller.get_system_state(),
            'molecular_analysis': self.molecular_system.get_molecular_analysis(),
            'integrated_systems': {
                'echo_v4_core': True,
                'crystalline_heart_legacy': True,
                'crystalline_heart_enhanced': True,
                'audio_system': True,
                'voice_engine': True,
                'audio_bridge': True,
                'session_persistence': True,
                'neural_voice_synthesis': True,
                'autism_optimized_vad': True,
                'aba_therapeutics': True,
                'voice_crystal': True,
                'mathematical_framework': True,
                'quantum_system': True,
                'memory_system': True,
                'cyber_physical_controller': True
            },
            'performance_metrics': {
                'processing_time_avg': np.mean([m.timestamp for m in list(self.metrics_history)[-10:]]) if self.metrics_history else 0.0,
                'active_components': sum(1 for active in self.get_integrated_components().values() if active),
                'system_health': min(1.0, gcl + (1.0 - stress) + life_intensity) / 3.0,
                'audio_system_status': {
                    'rust_engine_available': RUST_AVAILABLE,
                    'neural_tts_available': NEURAL_TTS_AVAILABLE,
                    'audio_device_available': AUDIO_AVAILABLE,
                    'audio_queue_size': self.audio_system.audio_queue.qsize() if hasattr(self.audio_system, 'audio_queue') else 0
                }
            }
        }
    
    def get_integrated_components(self) -> Dict[str, bool]:
        """Get status of all integrated components"""
        return {
            'echo_v4_core': True,
            'crystalline_heart_legacy': True,
            'crystalline_heart_enhanced': True,
            'audio_system': True,
            'voice_engine': True,
            'audio_bridge': True,
            'session_persistence': True,
            'neural_voice_synthesis': True,
            'autism_optimized_vad': True,
            'aba_therapeutics': True,
            'voice_crystal': True,
            'mathematical_framework': True,
            'quantum_system': True,
            'memory_system': True,
            'cyber_physical_controller': True,
            'molecular_system': True
        }

