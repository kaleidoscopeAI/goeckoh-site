"""
Cognitive Nebula Integration for Bubble Therapeutic System

Integrates 3D AI visualization with voice therapy to create
engaging therapeutic games for TBI and autism patients.

Core Purpose: Transform speech practice into creative universe exploration
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class VoicePrompt:
    """Therapeutic voice prompt for universe creation"""
    user_speech: str
    corrected_speech: str
    confidence: float
    therapeutic_goal: str
    universe_concept: str

@dataclass
class UniverseState:
    """3D universe state based on voice input"""
    node_count: int
    color_scheme: Dict[str, float]
    particle_behavior: str
    sound_reactivity: float
    difficulty_level: int

class CognitiveNebulaTherapist:
    """Therapeutic integration controller"""
    
    def __init__(self, bubble_system):
        self.bubble = bubble_system
        self.nebula_path = Path(__file__).parent.parent / "cognitive-nebula"
        self.current_session: Optional[Dict[str, Any]] = None
        self.therapeutic_progress: Dict[str, float] = {}
        
    def initialize_session(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize therapeutic gaming session"""
        
        # Assess user's therapeutic needs
        needs_assessment = self._assess_therapeutic_needs(user_profile)
        
        # Create personalized universe configuration
        universe_config = self._create_universe_config(needs_assessment)
        
        # Set up speech-to-universe mapping
        speech_mapping = self._create_speech_mapping(needs_assessment)
        
        self.current_session = {
            "user_profile": user_profile,
            "needs_assessment": needs_assessment,
            "universe_config": universe_config,
            "speech_mapping": speech_mapping,
            "start_time": datetime.now(),
            "voice_prompts": [],
            "progress_metrics": {
                "speech_attempts": 0,
                "successful_corrections": 0,
                "creativity_score": 0.0,
                "engagement_time": 0.0
            }
        }
        
        logger.info(f"Initialized therapeutic session for {user_profile.get('name', 'User')}")
        return self.current_session
    
    def _assess_therapeutic_needs(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Assess user's therapeutic needs based on profile"""
        
        # Speech therapy needs
        speech_needs = {
            "articulation_practice": user_profile.get("articulation_difficulty", 0.5),
            "fluency_building": user_profile.get("fluency_issues", 0.5),
            "vocabulary_expansion": user_profile.get("limited_vocabulary", 0.5),
            "social_communication": user_profile.get("social_difficulty", 0.5)
        }
        
        # Cognitive needs
        cognitive_needs = {
            "creativity_encouragement": user_profile.get("creativity_block", 0.5),
            "focus_improvement": user_profile.get("attention_issues", 0.5),
            "emotional_expression": user_profile.get("emotional_difficulty", 0.5),
            "sensory_processing": user_profile.get("sensory_sensitivity", 0.5)
        }
        
        # Determine primary focus areas
        primary_needs = []
        for need, score in {**speech_needs, **cognitive_needs}.items():
            if score > 0.6:
                primary_needs.append(need)
        
        return {
            "speech_needs": speech_needs,
            "cognitive_needs": cognitive_needs,
            "primary_focus": primary_needs,
            "difficulty_level": self._calculate_difficulty_level(user_profile),
            "motivation_type": self._determine_motivation_type(user_profile)
        }
    
    def _calculate_difficulty_level(self, user_profile: Dict[str, Any]) -> int:
        """Calculate appropriate difficulty level (1-10)"""
        base_difficulty = user_profile.get("age", 25) // 10  # Age-based
        
        # Adjust for speech ability
        speech_factor = 1.0 - user_profile.get("speech_ability", 0.5)
        
        # Adjust for cognitive load tolerance
        cognitive_tolerance = user_profile.get("cognitive_tolerance", 0.5)
        
        difficulty = int(base_difficulty + (speech_factor * 5) - (cognitive_tolerance * 3))
        return max(1, min(10, difficulty))
    
    def _determine_motivation_type(self, user_profile: Dict[str, Any]) -> str:
        """Determine best motivation approach"""
        if user_profile.get("visual_preference", 0.5) > 0.7:
            return "visual_rewards"
        elif user_profile.get("competitive", 0.5) > 0.7:
            return "achievement_unlocking"
        elif user_profile.get("creative", 0.5) > 0.7:
            return "creative_expression"
        else:
            return "exploration_discovery"
    
    def _create_universe_config(self, needs: Dict[str, Any]) -> Dict[str, Any]:
        """Create universe configuration based on therapeutic needs"""
        
        # Base configuration
        config = {
            "particle_count": 15000,  # Start moderate
            "color_saturation": 1.0,
            "movement_speed": 0.8,
            "sound_reactivity": 0.7,
            "visual_complexity": 0.6
        }
        
        # Adjust for sensory sensitivity
        if needs["cognitive_needs"]["sensory_processing"] > 0.7:
            config.update({
                "particle_count": 8000,  # Reduce visual load
                "movement_speed": 0.5,
                "color_saturation": 0.7,
                "visual_complexity": 0.4
            })
        
        # Adjust for attention needs
        if needs["cognitive_needs"]["focus_improvement"] > 0.7:
            config.update({
                "sound_reactivity": 0.9,  # High audio-visual connection
                "movement_speed": 0.6,    # Calmer movement
                "particle_count": 12000
            })
        
        # Adjust for creativity needs
        if needs["cognitive_needs"]["creativity_encouragement"] > 0.7:
            config.update({
                "particle_count": 25000,  # More creative possibilities
                "color_saturation": 1.2,
                "visual_complexity": 0.8
            })
        
        return config
    
    def _create_speech_mapping(self, needs: Dict[str, Any]) -> Dict[str, Any]:
        """Create speech-to-universe effect mappings"""
        
        mappings = {
            "vocabulary_words": {
                "colors": ["red", "blue", "green", "yellow", "purple", "orange"],
                "shapes": ["circle", "square", "triangle", "star", "spiral"],
                "motions": ["spin", "float", "dance", "pulse", "flow"],
                "emotions": ["happy", "calm", "excited", "peaceful", "energetic"]
            },
            "speech_effects": {
                "volume": "particle_size",
                "pitch": "color_hue",
                "rhythm": "movement_speed",
                "clarity": "visual_focus"
            },
            "therapeutic_targets": []
        }
        
        # Add therapeutic target words based on needs
        if needs["speech_needs"]["articulation_practice"] > 0.6:
            mappings["therapeutic_targets"].extend([
                "rainbow", "butterfly", "waterfall", "mountain", "crystal"
            ])
        
        if needs["speech_needs"]["vocabulary_expansion"] > 0.6:
            mappings["therapeutic_targets"].extend([
                "galaxy", "nebula", "constellation", "asteroid", "universe"
            ])
        
        return mappings
    
    def process_voice_input(self, voice_data: Dict[str, Any]) -> VoicePrompt:
        """Process voice input and create therapeutic prompt"""
        
        raw_speech = voice_data.get("text", "")
        confidence = voice_data.get("confidence", 0.0)
        
        # Get corrected speech from Bubble's grammar system
        corrected_speech = self.bubble.correct_speech(raw_speech)
        
        # Determine therapeutic goal
        therapeutic_goal = self._identify_therapeutic_goal(raw_speech, corrected_speech)
        
        # Generate universe concept
        universe_concept = self._generate_universe_concept(corrected_speech, therapeutic_goal)
        
        prompt = VoicePrompt(
            user_speech=raw_speech,
            corrected_speech=corrected_speech,
            confidence=confidence,
            therapeutic_goal=therapeutic_goal,
            universe_concept=universe_concept
        )
        
        # Update session progress
        if self.current_session:
            self.current_session["voice_prompts"].append(prompt)
            self.current_session["progress_metrics"]["speech_attempts"] += 1
            
            if raw_speech != corrected_speech:
                self.current_session["progress_metrics"]["successful_corrections"] += 1
        
        return prompt
    
    def _identify_therapeutic_goal(self, raw: str, corrected: str) -> str:
        """Identify the therapeutic goal based on speech input"""
        
        # Check for specific therapeutic targets
        mappings = self.current_session["speech_mapping"] if self.current_session else {}
        targets = mappings.get("therapeutic_targets", [])
        
        for target in targets:
            if target.lower() in corrected.lower():
                return f"Practice articulation of '{target}'"
        
        # Check for vocabulary words
        vocab_categories = mappings.get("vocabulary_words", {})
        for category, words in vocab_categories.items():
            for word in words:
                if word.lower() in corrected.lower():
                    return f"Expand {category} vocabulary"
        
        # Check for speech corrections
        if raw != corrected:
            return "Improve speech clarity and pronunciation"
        
        return "Creative expression and communication"
    
    def _generate_universe_concept(self, speech: str, goal: str) -> str:
        """Generate universe concept based on speech and therapeutic goal"""
        
        # Extract key concepts from speech
        concepts = self._extract_visual_concepts(speech)
        
        # Create universe description
        if "color" in goal.lower():
            return f"A {concepts.get('color', 'rainbow')} universe with {concepts.get('motion', 'flowing')} particles"
        elif "shape" in goal.lower():
            return f"Universe filled with {concepts.get('shape', 'crystalline')} {concepts.get('objects', 'structures')}"
        elif "emotion" in goal.lower():
            return f"A {concepts.get('emotion', 'peaceful')} cosmic environment with {concepts.get('energy', 'gentle')} movements"
        else:
            return f"An imaginative {concepts.get('setting', 'cosmic')} realm with {concepts.get('elements', 'stellar')} features"
    
    def _extract_visual_concepts(self, speech: str) -> Dict[str, str]:
        """Extract visual concepts from speech"""
        
        concepts = {}
        speech_lower = speech.lower()
        
        # Color concepts
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black"]
        for color in colors:
            if color in speech_lower:
                concepts["color"] = color
                break
        
        # Shape concepts
        shapes = ["circle", "square", "triangle", "star", "spiral", "crystal", "sphere", "cube"]
        for shape in shapes:
            if shape in speech_lower:
                concepts["shape"] = shape
                break
        
        # Motion concepts
        motions = ["spin", "rotate", "float", "dance", "pulse", "flow", "wave", "orbit"]
        for motion in motions:
            if motion in speech_lower:
                concepts["motion"] = motion
                break
        
        # Emotion concepts
        emotions = ["happy", "sad", "angry", "excited", "calm", "peaceful", "energetic", "gentle"]
        for emotion in emotions:
            if emotion in speech_lower:
                concepts["emotion"] = emotion
                break
        
        # Setting concepts
        settings = ["space", "ocean", "forest", "mountain", "city", "garden", "cosmic", "dream"]
        for setting in settings:
            if setting in speech_lower:
                concepts["setting"] = setting
                break
        
        # Element concepts
        elements = ["stars", "planets", "galaxies", "nebula", "water", "fire", "earth", "air"]
        for element in elements:
            if element in speech_lower:
                concepts["elements"] = element
                break
        
        return concepts
    
    def generate_universe_parameters(self, prompt: VoicePrompt) -> Dict[str, Any]:
        """Generate universe parameters for Cognitive Nebula"""
        
        base_config = self.current_session["universe_config"] if self.current_session else {}
        
        # Modify based on speech content
        concepts = self._extract_visual_concepts(prompt.corrected_speech)
        
        params = base_config.copy()
        
        # Adjust particle count based on speech complexity
        word_count = len(prompt.corrected_speech.split())
        params["particle_count"] = int(10000 + word_count * 1000)
        params["particle_count"] = min(params["particle_count"], 30000)
        
        # Adjust colors based on concepts
        if "color" in concepts:
            color_hue = self._color_to_hue(concepts["color"])
            params["base_color_hue"] = color_hue
        
        # Adjust movement based on emotion
        if "emotion" in concepts:
            params["movement_speed"] = self._emotion_to_speed(concepts["emotion"])
        
        # Add visual effects based on therapeutic goal
        if "articulation" in prompt.therapeutic_goal:
            params["visual_focus"] = 0.8  # High focus on articulation targets
            params["particle_animation"] = "pulse"
        
        elif "vocabulary" in prompt.therapeutic_goal:
            params["visual_complexity"] = 0.8  # Rich visual vocabulary
            params["particle_animation"] = "morph"
        
        else:
            params["particle_animation"] = "flow"
        
        # Set AI generation prompt
        params["ai_prompt"] = self._create_ai_generation_prompt(prompt, concepts)
        
        return params
    
    def _color_to_hue(self, color: str) -> float:
        """Convert color name to hue value (0-1)"""
        color_map = {
            "red": 0.0, "orange": 0.08, "yellow": 0.17,
            "green": 0.33, "blue": 0.58, "purple": 0.75,
            "pink": 0.83, "white": 0.0, "black": 0.0
        }
        return color_map.get(color, 0.5)
    
    def _emotion_to_speed(self, emotion: str) -> float:
        """Convert emotion to movement speed"""
        emotion_map = {
            "energetic": 1.2, "excited": 1.1, "happy": 1.0,
            "calm": 0.6, "peaceful": 0.5, "gentle": 0.4,
            "sad": 0.3, "angry": 1.3
        }
        return emotion_map.get(emotion, 0.8)
    
    def _create_ai_generation_prompt(self, prompt: VoicePrompt, concepts: Dict[str, str]) -> str:
        """Create AI prompt for image generation"""
        
        base_prompt = f"Beautiful cosmic nebula featuring {concepts.get('setting', 'space')} environment"
        
        if "color" in concepts:
            base_prompt += f" with dominant {concepts['color']} colors"
        
        if "elements" in concepts:
            base_prompt += f" filled with {concepts['elements']}"
        
        if "emotion" in concepts:
            base_prompt += f", {concepts['emotion']} atmosphere"
        
        if "motion" in concepts:
            base_prompt += f", {concepts['motion']} movement"
        
        # Add therapeutic elements
        if "articulation" in prompt.therapeutic_goal:
            base_prompt += ", clear focused crystal structures"
        elif "vocabulary" in prompt.therapeutic_goal:
            base_prompt += ", diverse geometric patterns"
        
        return base_prompt + ", ethereal, magical, high quality, fantasy art"
    
    def update_progress_metrics(self, engagement_time: float, creativity_score: float):
        """Update session progress metrics"""
        if self.current_session:
            self.current_session["progress_metrics"]["engagement_time"] += engagement_time
            self.current_session["progress_metrics"]["creativity_score"] = creativity_score
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get therapeutic session summary"""
        if not self.current_session:
            return {}
        
        metrics = self.current_session["progress_metrics"]
        
        # Calculate success rates
        if metrics["speech_attempts"] > 0:
            correction_rate = metrics["successful_corrections"] / metrics["speech_attempts"]
        else:
            correction_rate = 0.0
        
        # Calculate engagement score
        engagement_score = min(1.0, metrics["engagement_time"] / 300.0)  # 5 minutes = full engagement
        
        return {
            "session_duration": metrics["engagement_time"],
            "speech_attempts": metrics["speech_attempts"],
            "correction_rate": correction_rate,
            "creativity_score": metrics["creativity_score"],
            "engagement_score": engagement_score,
            "therapeutic_goals_achieved": self._assess_goal_achievement(),
            "recommendations": self._generate_recommendations()
        }
    
    def _assess_goal_achievement(self) -> List[str]:
        """Assess which therapeutic goals were achieved"""
        achieved = []
        
        if self.current_session["progress_metrics"]["speech_attempts"] >= 10:
            achieved.append("speech_practice_completed")
        
        if self.current_session["progress_metrics"]["creativity_score"] > 0.7:
            achieved.append("creativity_encouraged")
        
        if self.current_session["progress_metrics"]["engagement_score"] > 0.8:
            achieved.append("high_engagement_achieved")
        
        return achieved
    
    def _generate_recommendations(self) -> List[str]:
        """Generate therapeutic recommendations"""
        recommendations = []
        
        metrics = self.current_session["progress_metrics"]
        
        if metrics["speech_attempts"] < 5:
            recommendations.append("Encourage more speech practice with simpler prompts")
        
        if metrics["creativity_score"] < 0.5:
            recommendations.append("Try more descriptive words to create richer universes")
        
        if metrics["engagement_time"] < 120:  # Less than 2 minutes
            recommendations.append("Consider shorter, more interactive sessions")
        
        return recommendations

# Integration interface for Bubble system
class CognitiveNebulaInterface:
    """Interface between Bubble system and Cognitive Nebula"""
    
    def __init__(self, bubble_system):
        self.therapist = CognitiveNebulaTherapist(bubble_system)
        self.nebula_process = None
        self.websocket_port = 8765
        
    async def start_nebula_session(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Start Cognitive Nebula therapeutic session"""
        
        # Initialize therapeutic session
        session = self.therapist.initialize_session(user_profile)
        
        # Launch Cognitive Nebula with therapeutic configuration
        await self._launch_nebula_with_config(session["universe_config"])
        
        return session
    
    async def process_voice_to_universe(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice input and update universe"""
        
        # Create therapeutic prompt
        prompt = self.therapist.process_voice_input(voice_data)
        
        # Generate universe parameters
        universe_params = self.therapist.generate_universe_parameters(prompt)
        
        # Send to Cognitive Nebula
        await self._send_to_nebula(universe_params)
        
        return {
            "prompt": prompt,
            "universe_params": universe_params,
            "therapeutic_goal": prompt.therapeutic_goal
        }
    
    async def _launch_nebula_with_config(self, config: Dict[str, Any]):
        """Launch Cognitive Nebula with therapeutic configuration"""
        # Implementation would start the nebula process with config
        pass
    
    async def _send_to_nebula(self, params: Dict[str, Any]):
        """Send parameters to running Cognitive Nebula"""
        # Implementation would send via WebSocket or IPC
        pass
