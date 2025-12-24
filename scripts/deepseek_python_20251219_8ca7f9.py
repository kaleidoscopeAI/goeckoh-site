"""
Psychoacoustic Visual Engine for Neuro-Acoustic Mirror
Implements Bouba-Kiki effect, visual healing, and sensorimotor feedback
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json
from enum import Enum

class VisualState(Enum):
    NORMAL = "normal"
    DISFLUENCY_DETECTED = "disfluency_detected"
    SPLICING_ACTIVE = "splicing_active"
    HEALING = "healing"
    SILENT = "silent"

@dataclass
class BubbleDNA:
    """Genetic blueprint for the visual bubble"""
    base_radius: float = 1.0
    base_sharpness: float = 0.3
    responsiveness: float = 0.15
    elasticity: float = 0.8
    color_palette: Tuple = (0.2, 0.6, 0.9)  # Primary color
    glow_intensity: float = 0.5

class VisualFoam:
    """
    Psychoacoustic visual bubble with:
    - Bouba-Kiki mapping (round vs sharp)
    - Visual healing during splicing
    - Sensorimotor feedback synchronization
    """
    
    def __init__(self, dna: BubbleDNA):
        self.dna = dna
        self.current_state = VisualState.NORMAL
        
        # Current visual properties
        self.radius = dna.base_radius
        self.sharpness = dna.base_sharpness
        self.color = dna.color_palette
        self.glow = 0.0
        self.velocity = 0.0
        self.target_radius = dna.base_radius
        
        # Healing state
        self.healing_progress = 0.0
        self.healing_duration = 0.5  # seconds
        self.splice_timestamp = 0.0
        
        # Smoothing filters
        self.energy_filter = ExponentialFilter(alpha=0.1)
        self.zcr_filter = ExponentialFilter(alpha=0.1)
        self.pitch_filter = ExponentialFilter(alpha=0.05)
        
        # History for visual momentum
        self.radius_history = []
        self.max_history = 10
        
    def update(self, features: Dict[str, Any], delta_time: float) -> Dict[str, Any]:
        """
        Update visual state based on audio features
        
        Args:
            features: Audio features from Rust kernel
            delta_time: Time since last frame
            
        Returns:
            Visual state dictionary
        """
        # Extract features
        energy = self.energy_filter.update(features.get('energy', 0.0))
        zcr = self.zcr_filter.update(features.get('zero_crossing_rate', 0.0))
        pitch = self.pitch_filter.update(features.get('pitch_estimate', 120.0))
        spectral_centroid = features.get('spectral_centroid', 0.0)
        is_disfluency = features.get('is_disfluency', False)
        is_voiced = features.get('is_voiced', False)
        
        # Update state machine
        self._update_state(is_disfluency, delta_time)
        
        # Apply state-specific visual rules
        if self.current_state == VisualState.SPLICING_ACTIVE:
            return self._splicing_visuals(delta_time)
        elif self.current_state == VisualState.HEALING:
            return self._healing_visuals(delta_time)
        elif self.current_state == VisualState.DISFLUENCY_DETECTED:
            return self._disfluency_visuals(energy, zcr, delta_time)
        else:
            return self._normal_visuals(energy, zcr, pitch, spectral_centroid, is_voiced, delta_time)
    
    def _update_state(self, is_disfluency: bool, delta_time: float):
        """Update visual state machine"""
        if self.current_state == VisualState.SPLICING_ACTIVE:
            self.healing_progress += delta_time / self.healing_duration
            if self.healing_progress >= 1.0:
                self.current_state = VisualState.HEALING
                self.healing_progress = 0.0
                
        elif self.current_state == VisualState.HEALING:
            self.healing_progress += delta_time / self.healing_duration
            if self.healing_progress >= 1.0:
                self.current_state = VisualState.NORMAL
                
        elif is_disfluency and self.current_state == VisualState.NORMAL:
            self.current_state = VisualState.DISFLUENCY_DETECTED
            
        elif not is_disfluency and self.current_state == VisualState.DISFLUENCY_DETECTED:
            self.current_state = VisualState.NORMAL
    
    def _normal_visuals(self, energy: float, zcr: float, pitch: float, 
                       spectral_centroid: float, is_voiced: bool, delta_time: float) -> Dict[str, Any]:
        """Generate visuals for normal speech"""
        
        # Bouba-Kiki mapping: high ZCR = sharp (Kiki), low ZCR = round (Bouba)
        target_sharpness = self._smoothstep(zcr, 0.05, 0.3)
        
        # Energy controls size (louder = bigger)
        size_multiplier = 1.0 + (energy * 2.5)
        
        # Pitch affects vertical position (higher pitch = higher)
        vertical_offset = np.interp(pitch, [80, 400], [-0.5, 0.5])
        
        # Spectral centroid affects color temperature
        # Low centroid (dull sounds) = cooler, high centroid (bright sounds) = warmer
        color_temperature = np.interp(spectral_centroid, [0, 5000], [0.0, 1.0])
        
        # Voicing affects opacity
        opacity = 0.7 + (0.3 if is_voiced else 0.0)
        
        # Calculate target radius with momentum
        self.target_radius = self.dna.base_radius * size_multiplier
        
        # Apply physics: spring-damper system for smooth movement
        force = (self.target_radius - self.radius) * self.dna.elasticity
        damping = self.velocity * 0.1
        acceleration = force - damping
        
        self.velocity += acceleration * delta_time * 60  # Scale to 60fps
        self.radius += self.velocity * delta_time * 60
        
        # Clamp radius
        self.radius = max(0.1, min(self.radius, 3.0))
        
        # Update history for momentum visualization
        self.radius_history.append(self.radius)
        if len(self.radius_history) > self.max_history:
            self.radius_history.pop(0)
        
        # Calculate color based on audio features
        hue = 0.6 - (zcr * 0.3)  # Blue for smooth, purple for sharp
        saturation = 0.7 + (energy * 0.3)
        value = 0.8 + (0.2 if is_voiced else 0.0)
        
        # Add subtle pulsing with speech rhythm
        pulse = 1.0 + (np.sin(self.radius * 10) * 0.05)
        
        return {
            'state': 'normal',
            'radius': self.radius * pulse,
            'sharpness': target_sharpness,
            'color': (hue, saturation, value, opacity),
            'glow': self.dna.glow_intensity * energy,
            'vertical_offset': vertical_offset,
            'momentum': self._calculate_momentum(),
            'velocity': self.velocity,
            'features': {
                'energy': energy,
                'zcr': zcr,
                'pitch': pitch,
                'is_voiced': is_voiced
            }
        }
    
    def _disfluency_visuals(self, energy: float, zcr: float, delta_time: float) -> Dict[str, Any]:
        """Visuals when disfluency is detected but not yet spliced"""
        # Create tension: bubble contracts and darkens
        tension = np.sin(time.time() * 10) * 0.5 + 0.5  # Pulsing tension
        
        target_radius = self.dna.base_radius * (0.8 + tension * 0.2)
        self.radius += (target_radius - self.radius) * self.dna.responsiveness
        
        # Darker, more saturated color
        hue = 0.0  # Red for alert
        saturation = 0.9
        value = 0.7
        
        # Increased sharpness (visual "edge")
        sharpness = min(1.0, self.sharpness + 0.3)
        
        return {
            'state': 'disfluency_detected',
            'radius': self.radius,
            'sharpness': sharpness,
            'color': (hue, saturation, value, 0.9),
            'glow': 0.8,
            'pulse_speed': 10.0,
            'tension': tension
        }
    
    def _splicing_visuals(self, delta_time: float) -> Dict[str, Any]:
        """Visuals during active splicing (the "Visual Lie")"""
        # During splicing, we maintain the bubble's size and appearance
        # This creates the illusion of continuity (the "healing glow")
        
        healing_phase = self.healing_progress
        
        # Maintain size (don't let it collapse)
        self.radius = max(self.radius, self.dna.base_radius * 1.2)
        
        # Healing glow effect
        glow_intensity = 0.7 + (np.sin(healing_phase * np.pi * 4) * 0.3)
        
        # Transition color from red (alert) to cyan (healing)
        start_color = (0.0, 0.9, 0.7)  # Red
        end_color = (0.5, 1.0, 1.0)   # Cyan
        
        t = healing_phase
        color = (
            start_color[0] * (1-t) + end_color[0] * t,
            start_color[1] * (1-t) + end_color[1] * t,
            start_color[2] * (1-t) + end_color[2] * t,
            1.0
        )
        
        # Slightly increased sharpness during healing
        sharpness = 0.4 + (0.3 * (1 - healing_phase))
        
        return {
            'state': 'splicing_active',
            'radius': self.radius,
            'sharpness': sharpness,
            'color': color,
            'glow': glow_intensity,
            'healing_progress': healing_phase,
            'healing_pulse': np.sin(healing_phase * np.pi * 8) * 0.5 + 0.5
        }
    
    def _healing_visuals(self, delta_time: float) -> Dict[str, Any]:
        """Visuals during post-splice healing"""
        healing_phase = self.healing_progress
        
        # Gradually return to normal
        target_radius = self.dna.base_radius
        self.radius += (target_radius - self.radius) * self.dna.responsiveness * 2
        
        # Gentle pulsing to indicate healing
        pulse = 1.0 + (np.sin(healing_phase * np.pi * 2) * 0.1)
        
        # Fade from cyan back to normal blue
        hue = 0.5 * (1 - healing_phase) + 0.6 * healing_phase
        saturation = 1.0 * (1 - healing_phase) + 0.7 * healing_phase
        
        return {
            'state': 'healing',
            'radius': self.radius * pulse,
            'sharpness': self.dna.base_sharpness,
            'color': (hue, saturation, 1.0, 0.9),
            'glow': 0.4 * (1 - healing_phase),
            'healing_progress': healing_phase
        }
    
    def trigger_splice(self):
        """Trigger visual healing sequence"""
        self.current_state = VisualState.SPLICING_ACTIVE
        self.healing_progress = 0.0
        self.splice_timestamp = time.time()
    
    def _smoothstep(self, x: float, edge0: float, edge1: float) -> float:
        """Smoothstep function for transitions"""
        x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return x * x * (3 - 2 * x)
    
    def _calculate_momentum(self) -> float:
        """Calculate visual momentum from radius history"""
        if len(self.radius_history) < 2:
            return 0.0
        
        # Calculate rate of change
        changes = np.diff(self.radius_history)
        return np.mean(changes) if len(changes) > 0 else 0.0

class ExponentialFilter:
    """Simple exponential smoothing filter"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0
    
    def update(self, new_value: float) -> float:
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class SensorimotorFeedback:
    """
    Maps audio features to haptic/visual feedback
    Implements the sensorimotor loop intervention
    """
    
    def __init__(self):
        self.feedback_gain = 1.0
        self.calibration = {
            'energy_to_amplitude': 0.5,
            'pitch_to_frequency': 0.01,
            'zcr_to_texture': 0.8
        }
    
    def generate_feedback(self, audio_features: Dict[str, Any], 
                         visual_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synchronized sensorimotor feedback
        
        Returns:
            Dict with haptic, visual, and timing data
        """
        energy = audio_features.get('energy', 0.0)
        zcr = audio_features.get('zero_crossing_rate', 0.0)
        pitch = audio_features.get('pitch_estimate', 120.0)
        
        # Haptic feedback (simulated - would control physical actuators)
        haptic = {
            'amplitude': energy * self.calibration['energy_to_amplitude'],
            'frequency': pitch * self.calibration['pitch_to_frequency'],
            'texture': zcr * self.calibration['zcr_to_texture'],
            'duration': 0.1  # seconds
        }
        
        # Timing synchronization
        sync = {
            'audio_latency': 0.25,  # 250ms lookahead
            'visual_latency': 0.016,  # 1 frame at 60fps
            'haptic_latency': 0.01,  # 10ms
            'is_synchronized': True
        }
        
        return {
            'haptic': haptic,
            'visual_amplification': self.feedback_gain,
            'synchronization': sync