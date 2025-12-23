class AbaProgress:
    """ABA skill progress tracking"""
    attempts: int = 0
    successes: int = 0
    last_attempt_ts: float = 0.0
    current_level: int = 1

class UnifiedAbaEngine:
    """Complete ABA therapeutics system"""
    
    def __init__(self):
        self.aba_skills = {
            "self_care": ["brush teeth", "wash hands", "dress self", "take medication"],
            "communication": ["greet others", "ask for help", "express feelings", "use AAC device"],
            "social_tom": ["share toys", "take turns", "understand emotions", "empathy response"]
        }
        
        self.rewards = [
            "Great job! You're getting better every day.",
            "Wow, that was awesome! High five!",
            "I'm so proud of you for trying that."
        ]
        
        self.progress = {}
        for cat, skills in self.aba_skills.items():
            self.progress[cat] = {skill: AbaProgress() for skill in skills}
        
        self.social_story_templates = {
            "transition": "Today, we're going from {current} to {next}. First, we say goodbye to {current}. Then, we walk calmly to {next}. It's okay to feel a little worried, but we'll have fun there!",
            "meltdown": "Sometimes we feel overwhelmed, like a storm inside. When that happens, we can take deep breaths: in for 4, out for 6. Or hug our favorite toy. Soon the storm passes, and we feel better.",
            "social": "When we see a friend, we can say 'Hi, want to play?' If they say yes, we share the toys. If no, that's okay – we can play next time."
        }
    
    def intervene(self, emotional_state: EmotionalState, text: Optional[str] = None) -> Dict[str, Any]:
        """ABA intervention based on emotional state and context"""
        intervention = {
            'strategy': None,
            'social_story': None,
            'reward': None,
            'skill_focus': None,
            'confidence': 0.0,
            'urgency': 'low'
        }
        
        # Calculate intervention confidence
        anxiety_score = emotional_state.anxiety + emotional_state.fear
        positive_score = emotional_state.joy + emotional_state.trust
        cognitive_load = emotional_state.overwhelm + (1.0 - emotional_state.focus)
        
        # High anxiety/fear → calming strategies
        if anxiety_score > 1.3:
            intervention['strategy'] = "calming"
            intervention['social_story'] = self.social_story_templates['meltdown']
            intervention['urgency'] = 'high'
            intervention['confidence'] = min(1.0, anxiety_score / 1.5)
            
            # Add specific calming techniques
            if emotional_state.overwhelm > 0.7:
                intervention['social_story'] += " Let's find a quiet space and use our calming strategies."
            
        # Low focus → attention strategies
        elif emotional_state.focus < 0.3 and cognitive_load > 0.8:
            intervention['strategy'] = "attention"
            intervention['skill_focus'] = "self_care"
            intervention['urgency'] = 'medium'
            intervention['confidence'] = min(1.0, (1.0 - emotional_state.focus) * 2)
            
            # Add attention-building activities with context awareness
            # Extract context from text if available
            context = text.lower() if text else ""
            
            if context:
                # Context-aware skill selection
                if "school" in context:
                    skill_options = ["raise hand", "listen to teacher", "follow instructions"]
                elif "home" in context:
                    skill_options = ["brush teeth", "get dressed", "eat breakfast"]
                else:
                    skill_options = self.aba_skills['self_care']
                
                selected_skill = np.random.choice(skill_options)
                intervention['social_story'] = f"Let's focus on one thing at a time. How about we try {selected_skill}?"
                intervention['skill_focus'] = selected_skill
            else:
                # Default attention activity
                intervention['social_story'] = f"Let's focus on one thing at a time. How about we try {np.random.choice(self.aba_skills['self_care'])}?"
        
        # High overwhelm → sensory regulation
        elif emotional_state.overwhelm > 0.6:
            intervention['strategy'] = "sensory"
            intervention['social_story'] = "Let's take a sensory break. Deep breaths: in for 4, out for 6. Let's reduce the stimulation around us."
            intervention['urgency'] = 'high'
            intervention['confidence'] = min(1.0, emotional_state.overwhelm * 1.2)
        
        # Positive states → reward and skill building with adaptive selection
        elif positive_score > 1.1:
            # Adaptive reward based on emotional state
            if emotional_state.joy > 0.8:
                # High joy - celebrate achievement
                intervention['reward'] = "Excellent work! You're doing absolutely amazing! 🎉"
            elif emotional_state.trust > 0.7:
                # High trust - reinforce social connection
                intervention['reward'] = "You're building such wonderful connections! Great job!"
            else:
                # General positive reinforcement
                intervention['reward'] = np.random.choice(self.rewards)
            
            # Skill selection based on current mastery levels
            skill_categories = list(self.aba_skills.keys())
            # Prioritize categories with lower average mastery
            category_mastery = {}
            for cat in skill_categories:
                if cat in self.progress:
                    avg_level = np.mean([prog.current_level for prog in self.progress[cat].values()])
                    category_mastery[cat] = avg_level
                else:
                    category_mastery[cat] = 1.0
            
            # Select category with lowest mastery (more room for growth)
            target_category = min(category_mastery.keys(), key=lambda k: category_mastery[k])
            available_skills = self.aba_skills[target_category]
            
            # Filter for skills that aren't already mastered
            available_skills = [skill for skill in available_skills 
                               if target_category in self.progress and 
                               skill in self.progress[target_category] and 
                               self.progress[target_category][skill].current_level < 5]
            
            if available_skills:
                intervention['skill_focus'] = np.random.choice(available_skills)
            else:
                intervention['skill_focus'] = np.random.choice(list(self.aba_skills['social_tom']))
            
            intervention['strategy'] = "reinforcement"
            intervention['confidence'] = min(1.0, positive_score / 1.5)
            
            # Add skill-specific encouragement with personalization
            skill = intervention['skill_focus']
            context = text.lower() if text else ""
            
            if "school" in context:
                intervention['social_story'] = f"You're doing great in school! Let's work on {skill} together to keep up this amazing progress."
            elif "home" in context:
                intervention['social_story'] = f"Wonderful work at home! Let's practice {skill} to make it even easier."
            else:
                intervention['social_story'] = f"You're doing great! Let's work on {skill} together."
        
        # Moderate distress → support strategies
        elif 0.5 < anxiety_score <= 1.3:
            intervention['strategy'] = "support"
            intervention['skill_focus'] = "communication"
            intervention['urgency'] = 'medium'
            intervention['confidence'] = 0.6
            intervention['social_story'] = "I'm here to help. Let's talk about what's bothering you."
        
        return intervention
    
    def track_skill_attempt(self, category: str, skill: str, success: bool, context: str = ""):
        """Track ABA skill progress with enhanced analytics"""
        if category in self.progress and skill in self.progress[category]:
            prog = self.progress[category][skill]
            prog.attempts += 1
            if success:
                prog.successes += 1
                prog.last_attempt_ts = time.time()
            
            # Enhanced level progression with context awareness
            success_rate = prog.successes / max(1, prog.attempts)
            
            # Adaptive progression thresholds
            if prog.current_level == 1:
                threshold = 0.7  # Easier to advance from level 1
                min_attempts = 3
            elif prog.current_level == 2:
                threshold = 0.8  # Standard progression
                min_attempts = 5
            else:
                threshold = 0.9  # Harder to reach mastery
                min_attempts = 10
            
            if success_rate > threshold and prog.attempts >= min_attempts:
                prog.current_level = min(3, prog.current_level + 1)
                
            # Track context patterns
            if hasattr(prog, 'contexts'):
                prog.contexts.append(context)
            else:
                prog.contexts = [context]
                
            # Calculate skill mastery score
            prog.mastery_score = self._calculate_mastery_score(prog)
    
    def _calculate_mastery_score(self, progress: AbaProgress) -> float:
        """Calculate comprehensive mastery score"""
        if progress.attempts == 0:
            return 0.0
            
        base_success_rate = progress.successes / progress.attempts
        level_bonus = progress.current_level * 0.1
        consistency_bonus = 0.0
        
        # Check for consistent performance
        if hasattr(progress, 'contexts') and len(progress.contexts) >= 5:
            recent_contexts = progress.contexts[-5:]
            if all('success' in ctx.lower() for ctx in recent_contexts):
                consistency_bonus = 0.2
                
        return min(1.0, base_success_rate + level_bonus + consistency_bonus)
    
    def get_success_rate(self) -> float:
        """Calculate overall ABA success rate with weighted scoring"""
        total_attempts = 0
        total_successes = 0
        category_weights = {
            "self_care": 1.2,      # Higher weight for self-care skills
            "communication": 1.5,  # Highest weight for communication
            "social_tom": 1.0       # Standard weight for social skills
        }
        
        weighted_successes = 0.0
        weighted_attempts = 0.0
        
        for cat_name, cat_skills in self.progress.items():
            weight = category_weights.get(cat_name, 1.0)
            for prog in cat_skills.values():
                weighted_successes += prog.successes * weight
                weighted_attempts += prog.attempts * weight
                total_successes += prog.successes
                total_attempts += prog.attempts
        
        # Return both raw and weighted success rates
        raw_rate = total_successes / max(1, total_attempts)
        weighted_rate = weighted_successes / max(1, weighted_attempts)
        
        return weighted_rate
    
    def get_detailed_progress(self) -> Dict[str, Any]:
        """Get detailed progress analytics"""
        analytics = {
            'overall_success_rate': self.get_success_rate(),
            'total_attempts': sum(prog.attempts for cat_skills in self.progress.values() for prog in cat_skills.values()),
            'category_performance': {},
            'skill_mastery_levels': {},
            'recent_activity': [],
            'improvement_trends': {}
        }
        
        # Category-level analytics
        for cat_name, cat_skills in self.progress.items():
            cat_attempts = sum(prog.attempts for prog in cat_skills.values())
            cat_successes = sum(prog.successes for prog in cat_skills.values())
            cat_success_rate = cat_successes / max(1, cat_attempts)
            
            analytics['category_performance'][cat_name] = {
                'success_rate': cat_success_rate,
                'total_attempts': cat_attempts,
                'average_level': np.mean([prog.current_level for prog in cat_skills.values()])
            }
        
        # Skill mastery tracking
        for cat_name, cat_skills in self.progress.items():
            analytics['skill_mastery_levels'][cat_name] = {
                skill: {
                    'level': prog.current_level,
                    'success_rate': prog.successes / max(1, prog.attempts),
                    'attempts': prog.attempts,
                    'mastery_score': getattr(prog, 'mastery_score', 0.0)
                }
                for skill, prog in cat_skills.items()
            }
        
        return analytics

class AutismOptimizedVAD:
    """Autism-optimized Voice Activity Detection"""
    
    def __init__(self):
        # Autism-tuned parameters
        self.threshold = 0.45
        self.min_silence_duration_ms = 1200
        self.speech_pad_ms = 400
        self.min_speech_duration_ms = 250
        self.sample_rate = 16000
        self.accumulated_speech_energy = 0.0
        self.speech_threshold = 2.0
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
        """Process audio with autism-optimized parameters"""
        # Calculate speech energy with autism-optimized sensitivity
        energy = np.sum(audio_chunk ** 2)
        self.accumulated_speech_energy += energy
        
        # Dynamic threshold adjustment for quiet/monotone speech
        dynamic_threshold = self.threshold * 1000 * (1.0 + 0.2 * np.sin(time.time() * 0.1))
        is_speech = energy > dynamic_threshold
        
        # Autism-optimized pause detection with longer tolerance
        if self.accumulated_speech_energy > self.speech_threshold:
            should_transcribe = True
            self.accumulated_speech_energy = 0.0
            
            # Log speech detection for autism analytics
            if hasattr(self, 'speech_detections'):
                self.speech_detections.append(time.time())
            else:
                self.speech_detections = [time.time()]
        else:
            should_transcribe = False
            
        return is_speech, should_transcribe
    
    def get_pause_analysis(self) -> Dict[str, float]:
        """Analyze speech patterns for autism support"""
        if not hasattr(self, 'speech_detections') or len(self.speech_detections) < 2:
            return {'avg_pause_duration': 0.0, 'speech_rate': 0.0, 'pause_variance': 0.0}
            
        # Calculate pause durations
        pauses = []
        for i in range(1, len(self.speech_detections)):
            pause_duration = self.speech_detections[i] - self.speech_detections[i-1]
            pauses.append(pause_duration)
            
        if pauses:
            avg_pause = np.mean(pauses)
            speech_rate = 1.0 / avg_pause if avg_pause > 0 else 0.0
            pause_variance = np.var(pauses)
            
            return {
                'avg_pause_duration': avg_pause,
                'speech_rate': speech_rate,
                'pause_variance': pause_variance,
                'long_pause_count': sum(1 for p in pauses if p > self.min_silence_duration_ms / 1000.0)
            }
        
        return {'avg_pause_duration': 0.0, 'speech_rate': 0.0, 'pause_variance': 0.0}

