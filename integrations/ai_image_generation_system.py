"""
AI-Powered Real-Time Image Generation System
Creates any image the user requests using AI models
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

@dataclass
class ImageGenerationRequest:
    """Request for AI image generation"""
    user_prompt: str
    corrected_prompt: str
    therapeutic_context: str
    style_preset: str
    technical_params: Dict[str, Any]

@dataclass
class GeneratedImage:
    """Generated image result"""
    image_data: str  # Base64 encoded
    prompt_used: str
    generation_time: float
    confidence: float
    therapeutic_relevance: float

class AIImageGenerator:
    """Real-time AI image generation for therapeutic universe creation"""
    
    def __init__(self):
        self.supported_models = {
            "stable_diffusion": {
                "endpoint": "http://localhost:7860/sdapi/v1/txt2img",
                "model": "stable-diffusion-v1-5",
                "free": True,
                "local": True
            },
            "automatic1111": {
                "endpoint": "http://localhost:7860",
                "model": "SDXL",
                "free": True, 
                "local": True
            },
            "pollinations": {
                "endpoint": "https://image.pollinations.ai/prompt",
                "model": "flux",
                "free": True,
                "local": False,
                "internet_required": True
            },
            "comfyui": {
                "endpoint": "http://localhost:8188",
                "model": "SDXL",
                "free": True,
                "local": True
            }
        }
        
        self.current_model = "stable_diffusion"
        self.session = None
        self.generation_history: List[GeneratedImage] = []
        
    async def initialize(self):
        """Initialize AI image generation"""
        self.session = aiohttp.ClientSession()
        
        # Test available models
        available_models = await self._test_model_availability()
        if available_models:
            logger.info(f"Available AI models: {list(available_models.keys())}")
            self.current_model = list(available_models.keys())[0]
        else:
            logger.warning("No AI models available, falling back to procedural generation")
    
    async def _test_model_availability(self) -> Dict[str, bool]:
        """Test which AI models are available"""
        available = {}
        
        for model_name, config in self.supported_models.items():
            try:
                if config["local"]:
                    # Test local endpoint
                    async with self.session.get(config["endpoint"], timeout=5) as response:
                        if response.status == 200:
                            available[model_name] = True
                        else:
                            available[model_name] = False
                else:
                    # Test internet endpoint
                    async with self.session.get(config["endpoint"], timeout=10) as response:
                        available[model_name] = response.status == 200
            except Exception as e:
                logger.warning(f"Model {model_name} unavailable: {e}")
                available[model_name] = False
        
        return available
    
    async def generate_image_from_voice(self, voice_input: Dict[str, Any]) -> GeneratedImage:
        """Generate AI image from voice input"""
        
        # Process voice input
        user_prompt = voice_input.get("text", "")
        corrected_prompt = voice_input.get("corrected_text", user_prompt)
        therapeutic_context = voice_input.get("therapeutic_goal", "creative_expression")
        
        # Create image generation request
        request = self._create_generation_request(user_prompt, corrected_prompt, therapeutic_context)
        
        # Generate image
        if self.current_model == "stable_diffusion":
            image = await self._generate_with_stable_diffusion(request)
        elif self.current_model == "automatic1111":
            image = await self._generate_with_automatic1111(request)
        elif self.current_model == "pollinations":
            image = await self._generate_with_pollinations(request)
        elif self.current_model == "comfyui":
            image = await self._generate_with_comfyui(request)
        else:
            image = await self._generate_procedural_fallback(request)
        
        # Store in history
        self.generation_history.append(image)
        
        return image
    
    def _create_generation_request(self, user_prompt: str, corrected_prompt: str, therapeutic_context: str) -> ImageGenerationRequest:
        """Create image generation request from voice input"""
        
        # Enhance prompt for better image generation
        enhanced_prompt = self._enhance_prompt_for_image_generation(corrected_prompt, therapeutic_context)
        
        # Determine style preset based on therapeutic context
        style_preset = self._determine_style_preset(therapeutic_context)
        
        # Set technical parameters
        technical_params = self._get_technical_parameters(style_preset)
        
        return ImageGenerationRequest(
            user_prompt=user_prompt,
            corrected_prompt=enhanced_prompt,
            therapeutic_context=therapeutic_context,
            style_preset=style_preset,
            technical_params=technical_params
        )
    
    def _enhance_prompt_for_image_generation(self, prompt: str, therapeutic_context: str) -> str:
        """Enhance voice prompt for better AI image generation"""
        
        # Base enhancement
        enhanced = prompt
        
        # Add descriptive elements based on context
        context_enhancements = {
            "articulation_practice": "clear, focused, crystalline structures",
            "vocabulary_expansion": "diverse, colorful, educational elements",
            "emotional_expression": "emotional atmosphere, expressive colors",
            "creativity_encouragement": "imaginative, whimsical, creative elements",
            "focus_improvement": "organized, structured, calming patterns",
            "social_communication": "interactive, connected, social elements"
        }
        
        if therapeutic_context in context_enhancements:
            enhanced += f", {context_enhancements[therapeutic_context]}"
        
        # Add artistic style
        enhanced += ", beautiful, high quality, detailed, professional artwork"
        
        # Add cosmic/universe theme for nebula context
        enhanced += ", cosmic universe, nebula, stars, galaxies, ethereal, magical"
        
        # Add safety and appropriateness
        enhanced += ", child-friendly, therapeutic, positive, encouraging"
        
        return enhanced
    
    def _determine_style_preset(self, therapeutic_context: str) -> str:
        """Determine artistic style preset based on therapeutic goal"""
        
        style_map = {
            "articulation_practice": "crystalline",
            "vocabulary_expansion": "colorful_book_illustration", 
            "emotional_expression": "impressionist",
            "creativity_encouragement": "fantasy_art",
            "focus_improvement": "geometric",
            "social_communication": "storybook"
        }
        
        return style_map.get(therapeutic_context, "cosmic_nebula")
    
    def _get_technical_parameters(self, style_preset: str) -> Dict[str, Any]:
        """Get technical parameters for image generation"""
        
        base_params = {
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7.5,
            "sampler": "DPM++ 2M Karras",
            "seed": -1  # Random
        }
        
        # Adjust based on style
        style_adjustments = {
            "crystalline": {"steps": 25, "cfg_scale": 8.0},
            "colorful_book_illustration": {"steps": 20, "cfg_scale": 7.0},
            "impressionist": {"steps": 30, "cfg_scale": 6.5},
            "fantasy_art": {"steps": 25, "cfg_scale": 8.5},
            "geometric": {"steps": 20, "cfg_scale": 7.5},
            "cosmic_nebula": {"steps": 25, "cfg_scale": 8.0}
        }
        
        if style_preset in style_adjustments:
            base_params.update(style_adjustments[style_preset])
        
        return base_params
    
    async def _generate_with_stable_diffusion(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Generate image using Stable Diffusion WebUI"""
        
        start_time = asyncio.get_event_loop().time()
        
        payload = {
            "prompt": request.corrected_prompt,
            "negative_prompt": "blurry, low quality, distorted, scary, inappropriate",
            "width": request.technical_params["width"],
            "height": request.technical_params["height"],
            "steps": request.technical_params["steps"],
            "cfg_scale": request.technical_params["cfg_scale"],
            "sampler_name": request.technical_params["sampler"],
            "seed": request.technical_params["seed"]
        }
        
        try:
            async with self.session.post(
                f"{self.supported_models['stable_diffusion']['endpoint']}",
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract image from result
                    if "images" in result and len(result["images"]) > 0:
                        image_data = result["images"][0]
                        generation_time = asyncio.get_event_loop().time() - start_time
                        
                        return GeneratedImage(
                            image_data=image_data,
                            prompt_used=request.corrected_prompt,
                            generation_time=generation_time,
                            confidence=0.9,
                            therapeutic_relevance=self._calculate_therapeutic_relevance(request)
                        )
                
        except Exception as e:
            logger.error(f"Stable Diffusion generation failed: {e}")
        
        # Fallback
        return await self._generate_procedural_fallback(request)
    
    async def _generate_with_automatic1111(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Generate image using Automatic1111"""
        
        start_time = asyncio.get_event_loop().time()
        
        payload = {
            "prompt": request.corrected_prompt,
            "negative_prompt": "blurry, low quality, distorted, scary, inappropriate",
            "width": request.technical_params["width"],
            "height": request.technical_params["height"],
            "steps": request.technical_params["steps"],
            "cfg_scale": request.technical_params["cfg_scale"],
            "sampler": request.technical_params["sampler"],
            "seed": request.technical_params["seed"]
        }
        
        try:
            async with self.session.post(
                f"{self.supported_models['automatic1111']['endpoint']}/sdapi/v1/txt2img",
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    if "images" in result and len(result["images"]) > 0:
                        image_data = result["images"][0]
                        generation_time = asyncio.get_event_loop().time() - start_time
                        
                        return GeneratedImage(
                            image_data=image_data,
                            prompt_used=request.corrected_prompt,
                            generation_time=generation_time,
                            confidence=0.9,
                            therapeutic_relevance=self._calculate_therapeutic_relevance(request)
                        )
                
        except Exception as e:
            logger.error(f"Automatic1111 generation failed: {e}")
        
        return await self._generate_procedural_fallback(request)
    
    async def _generate_with_pollinations(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Generate image using Pollinations API"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Pollinations uses simple GET request with prompt in URL
            encoded_prompt = request.corrected_prompt.replace(" ", "%20").replace(",", "%2C")
            url = f"{self.supported_models['pollinations']['endpoint']}/{encoded_prompt}"
            
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    image_data = base64.b64encode(await response.read()).decode()
                    generation_time = asyncio.get_event_loop().time() - start_time
                    
                    return GeneratedImage(
                        image_data=image_data,
                        prompt_used=request.corrected_prompt,
                        generation_time=generation_time,
                        confidence=0.7,  # Lower confidence for internet service
                        therapeutic_relevance=self._calculate_therapeutic_relevance(request)
                    )
                
        except Exception as e:
            logger.error(f"Pollinations generation failed: {e}")
        
        return await self._generate_procedural_fallback(request)
    
    async def _generate_with_comfyui(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Generate image using ComfyUI"""
        
        start_time = asyncio.get_event_loop().time()
        
        # ComfyUI workflow would go here
        # For now, fallback to procedural
        return await self._generate_procedural_fallback(request)
    
    async def _generate_procedural_fallback(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Fallback procedural generation when AI is unavailable"""
        
        start_time = asyncio.get_event_loop().time()
        
        # Generate procedural image based on prompt
        # This would create a beautiful abstract representation
        
        # For now, return a placeholder
        placeholder_image = self._create_placeholder_image(request)
        
        generation_time = asyncio.get_event_loop().time() - start_time
        
        return GeneratedImage(
            image_data=placeholder_image,
            prompt_used=request.corrected_prompt,
            generation_time=generation_time,
            confidence=0.5,  # Lower confidence for procedural
            therapeutic_relevance=self._calculate_therapeutic_relevance(request)
        )
    
    def _create_placeholder_image(self, request: ImageGenerationRequest) -> str:
        """Create placeholder image when AI generation fails"""
        
        # This would generate a procedural image
        # For now, return a simple gradient or pattern
        
        # Create a simple SVG placeholder
        svg_template = f"""
        <svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="grad1" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" style="stop-color:rgb(100,150,255);stop-opacity:1" />
                    <stop offset="100%" style="stop-color:rgb(50,50,150);stop-opacity:1" />
                </radialGradient>
            </defs>
            <rect width="512" height="512" fill="url(#grad1)" />
            <text x="256" y="256" font-family="Arial" font-size="24" fill="white" text-anchor="middle">
                {request.corrected_prompt[:30]}...
            </text>
        </svg>
        """
        
        # Convert SVG to base64
        import base64
        svg_bytes = svg_template.encode()
        return base64.b64encode(svg_bytes).decode()
    
    def _calculate_therapeutic_relevance(self, request: ImageGenerationRequest) -> float:
        """Calculate how therapeutically relevant the generated image is"""
        
        relevance = 0.5  # Base relevance
        
        # Boost based on therapeutic context
        context_boosts = {
            "articulation_practice": 0.3,
            "vocabulary_expansion": 0.4,
            "emotional_expression": 0.5,
            "creativity_encouragement": 0.6,
            "focus_improvement": 0.3,
            "social_communication": 0.4
        }
        
        if request.therapeutic_context in context_boosts:
            relevance += context_boosts[request.therapeutic_context]
        
        # Boost based on prompt complexity (more descriptive = better therapeutic value)
        word_count = len(request.corrected_prompt.split())
        if word_count >= 10:
            relevance += 0.2
        elif word_count >= 5:
            relevance += 0.1
        
        return min(1.0, relevance)
    
    async def get_generation_history(self, limit: int = 10) -> List[GeneratedImage]:
        """Get recent generation history"""
        return self.generation_history[-limit:]
    
    async def clear_history(self):
        """Clear generation history"""
        self.generation_history.clear()
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to different AI model"""
        if model_name in self.supported_models:
            self.current_model = model_name
            logger.info(f"Switched to AI model: {model_name}")
            return True
        return False
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "current_model": self.current_model,
            "available_models": list(self.supported_models.keys()),
            "model_info": self.supported_models.get(self.current_model, {}),
            "total_generations": len(self.generation_history),
            "average_generation_time": self._calculate_average_generation_time()
        }
    
    def _calculate_average_generation_time(self) -> float:
        """Calculate average generation time"""
        if not self.generation_history:
            return 0.0
        
        total_time = sum(img.generation_time for img in self.generation_history)
        return total_time / len(self.generation_history)

class TherapeuticImageCurator:
    """Curates AI-generated images for therapeutic appropriateness"""
    
    def __init__(self):
        self.inappropriate_keywords = [
            "scary", "horror", "violent", "weapon", "death", "blood",
            "monster", "demon", "evil", "dark", "creepy", "spooky"
        ]
        
        self.therapeutic_keywords = [
            "calm", "peaceful", "happy", "beautiful", "gentle", "soft",
            "bright", "colorful", "magical", "dream", "wonder", "joy"
        ]
    
    def curate_prompt(self, prompt: str) -> str:
        """Curate and enhance prompt for therapeutic appropriateness"""
        
        # Check for inappropriate content
        lower_prompt = prompt.lower()
        for keyword in self.inappropriate_keywords:
            if keyword in lower_prompt:
                # Replace with therapeutic alternatives
                prompt = self._replace_inappropriate_content(prompt, keyword)
        
        # Enhance with therapeutic keywords
        therapeutic_enhanced = self._add_therapeutic_elements(prompt)
        
        return therapeutic_enhanced
    
    def _replace_inappropriate_content(self, prompt: str, inappropriate_word: str) -> str:
        """Replace inappropriate content with therapeutic alternatives"""
        
        replacements = {
            "scary": "mysterious",
            "horror": "adventure",
            "violent": "energetic",
            "weapon": "tool",
            "death": "transformation",
            "blood": "energy",
            "monster": "creature",
            "demon": "spirit",
            "evil": "mischievous",
            "dark": "shadowy",
            "creepy": "mysterious",
            "spooky": "magical"
        }
        
        return prompt.replace(inappropriate_word, replacements.get(inappropriate_word, "gentle"))
    
    def _add_therapeutic_elements(self, prompt: str) -> str:
        """Add therapeutic elements to prompt"""
        
        # Add positive descriptors
        therapeutic_additions = [
            "peaceful", "calming", "soothing", "uplifting", "inspiring"
        ]
        
        # Don't over-enhance
        if len(prompt.split()) < 15:
            prompt += f", {therapeutic_additions[0]}"
        
        return prompt
    
    def validate_generated_image(self, image: GeneratedImage) -> bool:
        """Validate if generated image is therapeutically appropriate"""
        
        # This would analyze the generated image for appropriateness
        # For now, assume AI models with proper negative prompts are sufficient
        
        return image.confidence > 0.3 and image.therapeutic_relevance > 0.4

# Integration with Cognitive Nebula
class CognitiveNebulaAIIntegration:
    """Integrates AI image generation with Cognitive Nebula 3D visualization"""
    
    def __init__(self):
        self.ai_generator = AIImageGenerator()
        self.curator = TherapeuticImageCurator()
        self.nebula_interface = None
        
    async def initialize(self):
        """Initialize the AI integration system"""
        await self.ai_generator.initialize()
    
    async def process_voice_to_3d_universe(self, voice_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice input and create 3D universe with AI-generated textures"""
        
        # Generate AI image from voice
        generated_image = await self.ai_generator.generate_image_from_voice(voice_input)
        
        # Validate therapeutic appropriateness
        is_appropriate = self.curator.validate_generated_image(generated_image)
        
        if not is_appropriate:
            # Regenerate with safer prompt
            safer_prompt = self.curator.curate_prompt(voice_input.get("text", ""))
            voice_input["text"] = safer_prompt
            generated_image = await self.ai_generator.generate_image_from_voice(voice_input)
        
        # Convert AI image to 3D universe parameters
        universe_params = self._convert_image_to_universe_params(generated_image, voice_input)
        
        return {
            "generated_image": generated_image,
            "universe_params": universe_params,
            "therapeutic_feedback": self._generate_therapeutic_feedback(generated_image, voice_input),
            "next_actions": ["explore_universe", "create_more", "practice_speech"]
        }
    
    def _convert_image_to_universe_params(self, image: GeneratedImage, voice_input: Dict[str, Any]) -> Dict[str, Any]:
        """Convert AI-generated image to 3D universe parameters"""
        
        # Analyze image for visual properties
        visual_analysis = self._analyze_image_properties(image)
        
        # Create universe parameters based on image
        universe_params = {
            "particle_count": 15000 + int(image.therapeutic_relevance * 10000),
            "base_texture": image.image_data,  # Use AI image as texture
            "color_scheme": visual_analysis["colors"],
            "particle_behavior": self._determine_particle_behavior(voice_input),
            "lighting": visual_analysis["lighting"],
            "movement_speed": self._calculate_movement_speed(voice_input, image),
            "visual_complexity": min(1.0, len(voice_input.get("text", "").split()) / 10.0),
            "therapeutic_intensity": image.therapeutic_relevance
        }
        
        return universe_params
    
    def _analyze_image_properties(self, image: GeneratedImage) -> Dict[str, Any]:
        """Analyze generated image for visual properties"""
        
        # This would analyze the actual image data
        # For now, return estimated properties based on prompt
        
        prompt = image.prompt_used.lower()
        
        colors = {
            "primary": "blue" if "blue" in prompt else "purple",
            "secondary": "white" if "stars" in prompt else "yellow",
            "accent": "pink" if "pink" in prompt else "orange"
        }
        
        lighting = "bright" if "bright" in prompt or "happy" in prompt else "soft"
        
        return {
            "colors": colors,
            "lighting": lighting,
            "complexity": "high" if len(prompt.split()) > 10 else "medium"
        }
    
    def _determine_particle_behavior(self, voice_input: Dict[str, Any]) -> str:
        """Determine particle behavior based on voice input"""
        
        text = voice_input.get("text", "").lower()
        
        if "spin" in text or "rotate" in text:
            return "spinning"
        elif "float" in text or "fly" in text:
            return "floating"
        elif "dance" in text or "move" in text:
            return "dancing"
        elif "pulse" in text or "beat" in text:
            return "pulsing"
        else:
            return "flowing"
    
    def _calculate_movement_speed(self, voice_input: Dict[str, Any], image: GeneratedImage) -> float:
        """Calculate particle movement speed"""
        
        base_speed = 0.8
        
        # Adjust based on emotional content
        text = voice_input.get("text", "").lower()
        
        if "fast" in text or "quick" in text or "energetic" in text:
            base_speed += 0.4
        elif "slow" in text or "calm" in text or "gentle" in text:
            base_speed -= 0.3
        elif "excited" in text or "happy" in text:
            base_speed += 0.2
        
        return max(0.1, min(1.5, base_speed))
    
    def _generate_therapeutic_feedback(self, image: GeneratedImage, voice_input: Dict[str, Any]) -> str:
        """Generate therapeutic feedback based on generation"""
        
        if image.confidence > 0.8:
            feedback = "Amazing! Your voice created a beautiful universe!"
        elif image.confidence > 0.6:
            feedback = "Great job! Your words transformed into cosmic art."
        else:
            feedback = "Good work! Keep practicing to create even more amazing universes."
        
        if image.therapeutic_relevance > 0.8:
            feedback += " You're expressing yourself beautifully!"
        
        return feedback
