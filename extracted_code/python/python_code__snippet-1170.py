class AbaProgress:
    """ABA skill progress tracking"""
    attempts: int = 0
    successes: int = 0
    last_attempt_ts: float = 0.0
    current_level: int = 1

class RobustAbaEngine:
    """Robust ABA Therapeutics without external dependencies"""
    
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
            'skill_focus': None
        }
        
        # High anxiety/fear → calming strategies
        if emotional_state.anxiety > 0.7 or emotional_state.fear > 0.6:
            intervention['strategy'] = "calming"
            intervention['social_story'] = self.social_story_templates['meltdown']
        
        # Low focus → attention strategies
        elif emotional_state.focus < 0.3:
            intervention['strategy'] = "attention"
            intervention['skill_focus'] = "self_care"
        
        # High overwhelm → sensory regulation
        elif emotional_state.overwhelm > 0.6:
            intervention['strategy'] = "sensory"
            intervention['social_story'] = "Let's take a sensory break. Deep breaths: in for 4, out for 6."
        
        # Positive states → reward and skill building
        elif emotional_state.joy > 0.6 and emotional_state.trust > 0.5:
            intervention['reward'] = np.random.choice(self.rewards)
            intervention['skill_focus'] = np.random.choice(list(self.aba_skills['social_tom']))
        
        return intervention
    
    def track_skill_attempt(self, category: str, skill: str, success: bool):
        """Track ABA skill progress"""
        if category in self.progress and skill in self.progress[category]:
            prog = self.progress[category][skill]
            prog.attempts += 1
            if success:
                prog.successes += 1
            prog.last_attempt_ts = time.time()
            
            # Level progression
            success_rate = prog.successes / max(1, prog.attempts)
            if success_rate > 0.8 and prog.attempts >= 5:
                prog.current_level = min(3, prog.current_level + 1)
    
    def get_success_rate(self) -> float:
        """Calculate overall ABA success rate"""
        total_attempts = 0
        total_successes = 0
        
        for cat_skills in self.progress.values():
            for prog in cat_skills.values():
                total_attempts += prog.attempts
                total_successes += prog.successes
        
        return total_successes / max(1, total_attempts)

