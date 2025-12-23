"""
Voice-Powered Universe Game
Therapeutic game interface connecting speech practice to 3D universe creation
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class GameState(Enum):
    """Game states for therapeutic progression"""
    WELCOME = "welcome"
    VOICE_PROMPT = "voice_prompt"
    UNIVERSE_CREATION = "universe_creation"
    EXPLORATION = "exploration"
    THERAPEUTIC_FEEDBACK = "therapeutic_feedback"
    PROGRESSION = "progression"

@dataclass
class GameLevel:
    """Therapeutic game level configuration"""
    level_id: int
    name: str
    therapeutic_focus: str
    speech_targets: List[str]
    universe_theme: str
    difficulty: int
    rewards: List[str]

class VoiceUniverseGame:
    """Main therapeutic game controller"""
    
    def __init__(self, cognitive_nebula_interface):
        self.nebula = cognitive_nebula_interface
        self.current_state = GameState.WELCOME
        self.current_level: Optional[GameLevel] = None
        self.session_data: Dict[str, Any] = {}
        self.achievements: List[str] = []
        
        # Define therapeutic game levels
        self.levels = self._create_therapeutic_levels()
    
    def _create_therapeutic_levels(self) -> List[GameLevel]:
        """Create progressive therapeutic levels"""
        
        return [
            GameLevel(
                level_id=1,
                name="Color Explorer",
                therapeutic_focus="basic_color_vocabulary",
                speech_targets=["red", "blue", "green", "yellow", "purple"],
                universe_theme="colorful_nebula",
                difficulty=1,
                rewards=["unlock_new_colors", "praise_encouragement"]
            ),
            GameLevel(
                level_id=2,
                name="Shape Creator",
                therapeutic_focus="shape_vocabulary_articulation",
                speech_targets=["circle", "square", "triangle", "star", "heart"],
                universe_theme="geometric_universe",
                difficulty=2,
                rewards=["unlock_shapes", "visual_effects"]
            ),
            GameLevel(
                level_id=3,
                name="Emotion Painter",
                therapeutic_focus="emotional_expression",
                speech_targets=["happy", "calm", "excited", "peaceful", "love"],
                universe_theme="emotional_cosmos",
                difficulty=3,
                rewards=["emotion_particles", "music_integration"]
            ),
            GameLevel(
                level_id=4,
                name="Motion Master",
                therapeutic_focus="descriptive_language",
                speech_targets=["spinning", "floating", "dancing", "flowing", "pulsing"],
                universe_theme="dynamic_galaxy",
                difficulty=4,
                rewards=["complex_animations", "physics_simulation"]
            ),
            GameLevel(
                level_id=5,
                name="Story Builder",
                therapeutic_focus="narrative_construction",
                speech_targets=["adventure", "journey", "discovery", "creation", "magic"],
                universe_theme="story_universe",
                difficulty=5,
                rewards=["narrative_elements", "character_creation"]
            ),
            GameLevel(
                level_id=6,
                name="Universe Composer",
                therapeutic_focus="complex_creative_expression",
                speech_targets=["symphony", "harmony", "rhythm", "melody", "cosmos"],
                universe_theme="master_universe",
                difficulty=6,
                rewards=["full_customization", "sharing_capabilities"]
            )
        ]
    
    async def start_game_session(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Start therapeutic game session"""
        
        # Assess user and determine starting level
        starting_level = self._assess_starting_level(user_profile)
        self.current_level = starting_level
        
        # Initialize session data
        self.session_data = {
            "user_profile": user_profile,
            "start_time": time.time(),
            "current_level": starting_level.level_id,
            "completed_levels": [],
            "speech_attempts": [],
            "creativity_scores": [],
            "therapeutic_progress": {},
            "achievements": []
        }
        
        # Start with welcome state
        self.current_state = GameState.WELCOME
        
        # Initialize Cognitive Nebula
        await self.nebula.start_nebula_session(user_profile)
        
        logger.info(f"Started therapeutic game session at level {starting_level.level_id}")
        
        return {
            "game_state": self.current_state.value,
            "current_level": starting_level.name,
            "welcome_message": self._generate_welcome_message(starting_level),
            "first_prompt": self._generate_first_prompt(starting_level)
        }
    
    def _assess_starting_level(self, user_profile: Dict[str, Any]) -> GameLevel:
        """Assess user and determine appropriate starting level"""
        
        age = user_profile.get("age", 25)
        speech_ability = user_profile.get("speech_ability", 0.5)
        cognitive_level = user_profile.get("cognitive_level", 0.5)
        
        # Calculate starting level (1-6)
        base_level = max(1, min(6, int((age // 10) + (speech_ability * 3) + (cognitive_level * 2))))
        
        return self.levels[base_level - 1]
    
    def _generate_welcome_message(self, level: GameLevel) -> str:
        """Generate personalized welcome message"""
        
        messages = {
            1: f"Welcome to the Color Explorer! Let's create a beautiful universe using colors. Say the color names to paint your cosmos!",
            2: f"Welcome to Shape Creator! Your voice can now create shapes in space. Say shape names to build your geometric universe!",
            3: f"Welcome to Emotion Painter! Your feelings can create amazing cosmic environments. Express emotions to see them come alive!",
            4: f"Welcome to Motion Master! Your voice can control how things move in space. Describe movements to animate your universe!",
            5: f"Welcome to Story Builder! Your words can create entire worlds and adventures. Tell your cosmic story!",
            6: f"Welcome to Universe Composer! You're now a master creator. Combine everything to build your perfect universe!"
        }
        
        return messages.get(level.level_id, "Welcome to Voice Universe! Create with your voice!")
    
    def _generate_first_prompt(self, level: GameLevel) -> str:
        """Generate first therapeutic prompt"""
        
        prompts = {
            1: "Try saying 'blue' to create blue stars in your universe!",
            2: "Try saying 'circle' to create circular planets!",
            3: "Try saying 'happy' to create a happy, bright universe!",
            4: "Try saying 'spinning' to make things spin in space!",
            5: "Try saying 'adventure' to start your cosmic adventure!",
            6: "Try saying 'magic' to create a magical universe!"
        }
        
        return prompts.get(level.level_id, "Say something to create your universe!")
    
    async def process_voice_input(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice input and update game state"""
        
        # Determine action based on current state
        if self.current_state == GameState.VOICE_PROMPT:
            return await self._process_voice_prompt(voice_data)
        elif self.current_state == GameState.EXPLORATION:
            return await self._process_exploration_voice(voice_data)
        else:
            return await self._process_general_voice(voice_data)
    
    async def _process_voice_prompt(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice during prompt phase"""
        
        # Process through Cognitive Nebula
        universe_response = await self.nebula.process_voice_to_universe(voice_data)
        
        # Check if speech target was achieved
        target_achieved = self._check_speech_target(voice_data.get("text", ""))
        
        # Update game state
        if target_achieved:
            self.current_state = GameState.UNIVERSE_CREATION
            return {
                "game_action": "universe_creation",
                "universe_params": universe_response["universe_params"],
                "success_message": f"Great! You created {universe_response['prompt'].universe_concept}",
                "therapeutic_feedback": self._generate_therapeutic_feedback(target_achieved),
                "next_action": "explore"
            }
        else:
            return {
                "game_action": "encouragement",
                "message": "Keep trying! Say the target word clearly.",
                "hint": self._generate_hint(),
                "therapeutic_feedback": "Practice makes perfect! Try again."
            }
    
    async def _process_exploration_voice(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice during exploration phase"""
        
        universe_response = await self.nebula.process_voice_to_universe(voice_data)
        
        # Add creativity to universe
        return {
            "game_action": "universe_enhancement",
            "universe_params": universe_response["universe_params"],
            "message": f"You enhanced your universe with: {universe_response['prompt'].corrected_speech}",
            "creativity_bonus": self._calculate_creativity_bonus(voice_data),
            "exploration_progress": self._update_exploration_progress()
        }
    
    async def _process_general_voice(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process general voice input"""
        
        universe_response = await self.nebula.process_voice_to_universe(voice_data)
        
        return {
            "game_action": "universe_update",
            "universe_params": universe_response["universe_params"],
            "message": f"Your voice created: {universe_response['prompt'].universe_concept}"
        }
    
    def _check_speech_target(self, spoken_text: str) -> bool:
        """Check if speech target was achieved"""
        
        if not self.current_level:
            return False
        
        targets = self.current_level.speech_targets
        spoken_lower = spoken_text.lower().strip()
        
        for target in targets:
            if target.lower() in spoken_lower:
                # Record successful target
                self.session_data["speech_attempts"].append({
                    "text": spoken_text,
                    "target": target,
                    "success": True,
                    "timestamp": time.time()
                })
                return True
        
        # Record unsuccessful attempt
        self.session_data["speech_attempts"].append({
            "text": spoken_text,
            "target": targets[0] if targets else None,
            "success": False,
            "timestamp": time.time()
        })
        
        return False
    
    def _generate_therapeutic_feedback(self, target_achieved: bool) -> str:
        """Generate therapeutic feedback"""
        
        if target_achieved:
            feedback_messages = [
                "Excellent articulation! Your speech created something beautiful.",
                "Wonderful! Your voice is shaping your universe.",
                "Perfect! You're expressing yourself clearly and creatively.",
                "Amazing work! Your speech practice is paying off.",
                "Fantastic! You're building confidence with every word."
            ]
        else:
            feedback_messages = [
                "Keep practicing! Your voice is powerful.",
                "Almost there! Take a deep breath and try again.",
                "You're doing great! Let's try that word one more time.",
                "Good effort! Speech is a skill that improves with practice.",
                "You're learning! Every attempt makes you stronger."
            ]
        
        import random
        return random.choice(feedback_messages)
    
    def _generate_hint(self) -> str:
        """Generate hint for current level"""
        
        if not self.current_level:
            return "Try speaking clearly and slowly."
        
        hints = {
            1: "Try saying a color name like 'red' or 'blue'.",
            2: "Try saying a shape like 'circle' or 'star'.",
            3: "Try saying an emotion like 'happy' or 'calm'.",
            4: "Try saying a movement like 'spin' or 'float'.",
            5: "Try saying a story word like 'adventure' or 'magic'.",
            6: "Try combining words like 'magic spinning stars'."
        }
        
        return hints.get(self.current_level.level_id, "Speak clearly and expressively!")
    
    def _calculate_creativity_bonus(self, voice_data: Dict[str, Any]) -> float:
        """Calculate creativity bonus for exploration"""
        
        text = voice_data.get("text", "")
        word_count = len(text.split())
        
        # Bonus for longer, more descriptive speech
        if word_count >= 5:
            return 0.3
        elif word_count >= 3:
            return 0.2
        else:
            return 0.1
    
    def _update_exploration_progress(self) -> float:
        """Update exploration progress"""
        
        attempts = len(self.session_data.get("speech_attempts", []))
        successful = sum(1 for attempt in self.session_data.get("speech_attempts", []) if attempt.get("success", False))
        
        if attempts > 0:
            return successful / attempts
        return 0.0
    
    def check_level_completion(self) -> Optional[Dict[str, Any]]:
        """Check if current level is completed"""
        
        if not self.current_level:
            return None
        
        attempts = self.session_data.get("speech_attempts", [])
        successful_attempts = [a for a in attempts if a.get("success", False)]
        
        # Check completion criteria
        targets_needed = len(self.current_level.speech_targets)
        unique_targets_achieved = len(set(a.get("target") for a in successful_attempts if a.get("success", False)))
        
        if unique_targets_achieved >= targets_needed:
            return self._complete_level()
        
        return None
    
    def _complete_level(self) -> Dict[str, Any]:
        """Complete current level and prepare next"""
        
        # Record completion
        self.session_data["completed_levels"].append(self.current_level.level_id)
        
        # Calculate rewards
        rewards_earned = self._calculate_level_rewards()
        
        # Prepare next level
        next_level_id = self.current_level.level_id + 1
        next_level = None
        
        if next_level_id <= len(self.levels):
            next_level = self.levels[next_level_id - 1]
            self.current_level = next_level
            self.current_state = GameState.VOICE_PROMPT
        else:
            self.current_state = GameState.THERAPEUTIC_FEEDBACK
        
        return {
            "game_action": "level_complete",
            "completed_level": self.current_level.level_id,
            "rewards_earned": rewards_earned,
            "next_level": next_level.name if next_level else None,
            "completion_message": f"Congratulations! You completed {self.current_level.name}!",
            "therapeutic_achievements": self._assess_therapeutic_achievements()
        }
    
    def _calculate_level_rewards(self) -> List[str]:
        """Calculate rewards for level completion"""
        
        rewards = []
        
        if self.current_level:
            rewards.extend(self.current_level.rewards)
        
        # Add achievement-based rewards
        successful_rate = self._calculate_success_rate()
        
        if successful_rate >= 0.8:
            rewards.append("high_accuracy_achievement")
        
        if len(self.session_data.get("speech_attempts", [])) >= 10:
            rewards.append("persistent_practice_achievement")
        
        return rewards
    
    def _calculate_success_rate(self) -> float:
        """Calculate speech success rate"""
        
        attempts = self.session_data.get("speech_attempts", [])
        if not attempts:
            return 0.0
        
        successful = sum(1 for a in attempts if a.get("success", False))
        return successful / len(attempts)
    
    def _assess_therapeutic_achievements(self) -> Dict[str, Any]:
        """Assess therapeutic achievements"""
        
        return {
            "speech_improvement": self._calculate_success_rate(),
            "engagement_duration": time.time() - self.session_data.get("start_time", time.time()),
            "creativity_development": self._assess_creativity_development(),
            "confidence_building": self._assess_confidence_building(),
            "goals_achieved": self.session_data.get("completed_levels", [])
        }
    
    def _assess_creativity_development(self) -> float:
        """Assess creativity development"""
        
        attempts = self.session_data.get("speech_attempts", [])
        word_counts = [len(a.get("text", "").split()) for a in attempts]
        
        if not word_counts:
            return 0.0
        
        avg_word_count = sum(word_counts) / len(word_counts)
        return min(1.0, avg_word_count / 5.0)  # 5 words = full creativity
    
    def _assess_confidence_building(self) -> float:
        """Assess confidence building"""
        
        # Look at improvement over time
        attempts = self.session_data.get("speech_attempts", [])
        if len(attempts) < 3:
            return 0.0
        
        # Calculate improvement in success rate
        first_half = attempts[:len(attempts)//2]
        second_half = attempts[len(attempts)//2:]
        
        first_success_rate = sum(1 for a in first_half if a.get("success", False)) / len(first_half)
        second_success_rate = sum(1 for a in second_half if a.get("success", False)) / len(second_half)
        
        improvement = second_success_rate - first_success_rate
        return max(0.0, min(1.0, improvement + 0.5))  # Scale to 0-1
    
    def get_game_status(self) -> Dict[str, Any]:
        """Get current game status"""
        
        return {
            "current_state": self.current_state.value,
            "current_level": self.current_level.name if self.current_level else None,
            "progress": {
                "completed_levels": self.session_data.get("completed_levels", []),
                "success_rate": self._calculate_success_rate(),
                "engagement_time": time.time() - self.session_data.get("start_time", time.time()),
                "creativity_score": self._assess_creativity_development()
            },
            "achievements": self.achievements,
            "next_action": self._get_next_action()
        }
    
    def _get_next_action(self) -> str:
        """Get recommended next action"""
        
        if self.current_state == GameState.WELCOME:
            return "start_voice_prompt"
        elif self.current_state == GameState.VOICE_PROMPT:
            return "practice_speech_targets"
        elif self.current_state == GameState.UNIVERSE_CREATION:
            return "explore_universe"
        elif self.current_state == GameState.EXPLORATION:
            return "continue_exploring"
        elif self.current_state == GameState.THERAPEUTIC_FEEDBACK:
            return "review_progress"
        else:
            return "continue_playing"

# Game UI Controller
class GameUIController:
    """UI controller for the therapeutic game"""
    
    def __init__(self, game_engine: VoiceUniverseGame):
        self.game = game_engine
        self.ui_state = {}
    
    def render_game_screen(self) -> Dict[str, Any]:
        """Render current game screen"""
        
        status = self.game.get_game_status()
        
        if status["current_state"] == "welcome":
            return self._render_welcome_screen()
        elif status["current_state"] == "voice_prompt":
            return self._render_prompt_screen()
        elif status["current_state"] == "universe_creation":
            return self._render_creation_screen()
        elif status["current_state"] == "exploration":
            return self._render_exploration_screen()
        else:
            return self._render_default_screen()
    
    def _render_welcome_screen(self) -> Dict[str, Any]:
        """Render welcome screen"""
        
        level = self.game.current_level
        return {
            "screen_type": "welcome",
            "title": f"{level.name}" if self.game.current_level else "Voice Universe",
            "message": self.game._generate_welcome_message(self.game.current_level) if self.game.current_level else "Welcome!",
            "first_prompt": self.game._generate_first_prompt(self.game.current_level) if self.game.current_level else "Start creating!",
            "start_button": "Begin Your Journey",
            "visual_theme": "cosmic_welcome"
        }
    
    def _render_prompt_screen(self) -> Dict[str, Any]:
        """Render voice prompt screen"""
        
        return {
            "screen_type": "voice_prompt",
            "title": "Create With Your Voice!",
            "instruction": "Say the target word clearly to create your universe",
            "current_target": self.game.current_level.speech_targets[0] if self.game.current_level else "speak",
            "hint": self.game._generate_hint(),
            "visual_cues": ["microphone_active", "target_word_highlight"],
            "encouragement": "You can do this! Speak clearly and confidently."
        }
    
    def _render_creation_screen(self) -> Dict[str, Any]:
        """Render universe creation screen"""
        
        return {
            "screen_type": "universe_creation",
            "title": "Your Universe is Being Created!",
            "message": "Watch as your voice transforms into cosmic beauty",
            "visual_effects": ["particle_formation", "color_expansion", "shape_generation"],
            "progress_indicator": "creating",
            "next_action": "explore"
        }
    
    def _render_exploration_screen(self) -> Dict[str, Any]:
        """Render exploration screen"""
        
        return {
            "screen_type": "exploration",
            "title": "Explore Your Universe!",
            "message": "Keep speaking to enhance and expand your creation",
            "interactive_elements": ["voice_controls", "universe_navigation", "creativity_tools"],
            "progress": self.game.get_game_status()["progress"],
            "encouragement": "Amazing! Keep expressing yourself!"
        }
    
    def _render_default_screen(self) -> Dict[str, Any]:
        """Render default screen"""
        
        return {
            "screen_type": "default",
            "title": "Voice Universe",
            "message": "Continue your therapeutic journey",
            "status": self.game.get_game_status()
        }
