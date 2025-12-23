"""
UI Foam Widget for Kivy-based child interface

Provides real-time visual feedback based on acoustic features
with bubble physics and material properties.
"""

import numpy as np
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Mesh
from kivy.clock import Clock
from kivy.properties import NumericProperty, ListProperty
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FoamWidget(Widget):
    """Interactive bubble visualization widget for child interface"""
    
    # Physics properties
    rms_energy = NumericProperty(0.0)
    gcl_coherence = NumericProperty(1.0) 
    entropy = NumericProperty(0.0)
    
    # Visual properties
    bubble_color = ListProperty([0.5, 0.8, 1.0, 0.8])
    bubble_radius = NumericProperty(100.0)
    surface_roughness = NumericProperty(0.5)
    metallic_shine = NumericProperty(0.3)
    spikiness = NumericProperty(0.0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Physics state
        self.target_radius = 100.0
        self.target_color = [0.5, 0.8, 1.0, 0.8]
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        
        # Animation state
        self.breathing_phase = 0.0
        self.heartbeat_phase = 0.0
        self.pulse_intensity = 0.0
        
        # Visual mesh for bubble surface
        self.mesh_vertices = None
        self.mesh_indices = None
        
        # Initialize graphics
        self._init_graphics()
        
        # Start animation loop
        Clock.schedule_interval(self._update_physics, 1/60.0)
        Clock.schedule_interval(self._update_visuals, 1/30.0)
    
    def _init_graphics(self):
        """Initialize bubble graphics"""
        with self.canvas:
            # Clear existing instructions
            self.canvas.clear()
            
            # Create bubble mesh
            self._create_bubble_mesh()
            
            # Add glow effect
            Color(1.0, 1.0, 1.0, 0.1)
            self.glow_ellipse = Ellipse(
                pos=self.center_x - self.bubble_radius * 1.2,
                size=(self.bubble_radius * 2.4, self.bubble_radius * 2.4)
            )
            
            # Add main bubble
            Color(*self.bubble_color)
            self.main_ellipse = Ellipse(
                pos=self.center_x - self.bubble_radius,
                size=(self.bubble_radius * 2, self.bubble_radius * 2)
            )
            
            # Add surface details
            Color(1.0, 1.0, 1.0, 0.3)
            self.highlight = Ellipse(
                pos=self.center_x - self.bubble_radius * 0.3,
                size=(self.bubble_radius * 0.6, self.bubble_radius * 0.6)
            )
    
    def _create_bubble_mesh(self):
        """Create mesh vertices for bubble surface deformation"""
        # Create circular mesh with vertices for deformation
        num_vertices = 64
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        
        # Base circle vertices
        base_radius = self.bubble_radius
        self.mesh_vertices = []
        
        for angle in angles:
            x = np.cos(angle) * base_radius
            y = np.sin(angle) * base_radius
            self.mesh_vertices.extend([x + self.center_x, y + self.center_y, 0.0])
        
        # Add center vertex
        self.mesh_vertices.extend([self.center_x, self.center_y, 0.0])
        
        # Create triangle indices for mesh
        self.mesh_indices = []
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            # Triangle from center to edge
            self.mesh_indices.extend([i, next_i, num_vertices])
        
        # Create mesh if we have enough vertices
        if len(self.mesh_vertices) >= 9:  # Minimum 3 triangles
            try:
                self.bubble_mesh = Mesh(
                    vertices=self.mesh_vertices,
                    indices=self.mesh_indices,
                    fmt=[('v_pos', 3, 'float')]
                )
            except Exception as e:
                self.logger.warning(f"Failed to create mesh: {e}")
                self.bubble_mesh = None
    
    def update_physics(self, rms: float, gcl: float, entropy: float):
        """
        Update physics properties from audio analysis
        
        Args:
            rms: RMS energy from audio
            gcl: Global coherence level
            entropy: System entropy measure
        """
        self.rms_energy = rms
        self.gcl_coherence = gcl
        self.entropy = entropy
        
        # Update target radius based on energy
        energy_factor = np.clip(rms * 5.0, 0.5, 2.5)
        self.target_radius = 100.0 * energy_factor
        
        # Update color based on coherence
        if gcl > 0.7:
            # High coherence - calm blue
            self.target_color = [0.3, 0.6, 1.0, 0.8]
        elif gcl > 0.4:
            # Medium coherence - neutral
            self.target_color = [0.5, 0.8, 0.8, 0.8]
        else:
            # Low coherence - alert orange/red
            self.target_color = [1.0, 0.4, 0.2, 0.9]
        
        # Update material properties
        self.surface_roughness = 1.0 - gcl  # Low coherence = rough surface
        self.metallic_shine = entropy * 0.5  # High entropy = more metallic
        self.spikiness = np.clip((rms - 0.05) * 2.0, 0.0, 1.0)  # High energy = spiky
        
        # Trigger pulse for high energy
        if rms > 0.1:
            self.pulse_intensity = min(rms * 10.0, 1.0)
    
    def _update_physics(self, dt):
        """Update physics simulation"""
        # Breathing animation (always active)
        self.breathing_phase += dt * 2.0  # 2 Hz breathing
        breathing_factor = 0.05 * np.sin(self.breathing_phase)
        
        # Heartbeat animation (faster when stressed)
        heartbeat_rate = 1.5 + (1.0 - self.gcl_coherence) * 2.0  # 1.5-3.5 Hz
        self.heartbeat_phase += dt * heartbeat_rate * 2 * np.pi
        heartbeat_factor = 0.02 * np.sin(self.heartbeat_phase) * self.pulse_intensity
        
        # Smooth radius transitions
        radius_diff = self.target_radius - self.bubble_radius
        self.bubble_radius += radius_diff * 0.1  # Smooth interpolation
        
        # Apply animations
        animated_radius = self.bubble_radius * (1.0 + breathing_factor + heartbeat_factor)
        
        # Update bubble size
        self.bubble_radius = animated_radius
        
        # Decay pulse intensity
        self.pulse_intensity *= 0.95
    
    def _update_visuals(self, dt):
        """Update visual representation"""
        # Smooth color transitions
        for i in range(4):  # RGBA
            color_diff = self.target_color[i] - self.bubble_color[i]
            self.bubble_color[i] += color_diff * 0.1
        
        # Update graphics
        with self.canvas:
            # Clear and redraw
            self.canvas.clear()
            
            # Glow effect (varies with energy)
            glow_size = self.bubble_radius * 2.4 * (1.0 + self.rms_energy * 0.5)
            glow_alpha = 0.1 + self.rms_energy * 0.2
            Color(1.0, 1.0, 1.0, glow_alpha)
            self.glow_ellipse = Ellipse(
                pos=(self.center_x - glow_size/2, self.center_y - glow_size/2),
                size=(glow_size, glow_size)
            )
            
            # Main bubble
            Color(*self.bubble_color)
            self.main_ellipse = Ellipse(
                pos=(self.center_x - self.bubble_radius, self.center_y - self.bubble_radius),
                size=(self.bubble_radius * 2, self.bubble_radius * 2)
            )
            
            # Surface highlight (moves with breathing)
            highlight_offset = 0.3 * self.bubble_radius * np.sin(self.breathing_phase)
            highlight_size = self.bubble_radius * 0.4 * (1.0 + self.metallic_shine)
            Color(1.0, 1.0, 1.0, 0.3 + self.metallic_shine * 0.2)
            self.highlight = Ellipse(
                pos=(self.center_x - highlight_size/2 + highlight_offset, 
                     self.center_y - highlight_size/2),
                size=(highlight_size, highlight_size)
            )
            
            # Add spiky details if needed
            if self.spikiness > 0.3:
                self._add_spiky_details()
    
    def _add_spiky_details(self):
        """Add spiky surface details for high energy states"""
        num_spikes = int(8 * self.spikiness)
        spike_length = self.bubble_radius * 0.2 * self.spikiness
        
        Color(1.0, 1.0, 1.0, 0.5 * self.spikiness)
        
        for i in range(num_spikes):
            angle = (i / num_spikes) * 2 * np.pi + self.heartbeat_phase
            x = self.center_x + np.cos(angle) * (self.bubble_radius + spike_length)
            y = self.center_y + np.sin(angle) * (self.bubble_radius + spike_length)
            
            # Small spike circles
            Ellipse(pos=(x - 3, y - 3), size=(6, 6))
    
    def on_touch_down(self, touch):
        """Handle touch interactions"""
        if self.collide_point(*touch.pos):
            # Create ripple effect at touch point
            self._create_ripple(touch.pos)
            return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        """Handle touch movement"""
        if self.collide_point(*touch.pos):
            # Move bubble center slightly toward touch
            dx = touch.pos[0] - self.center_x
            dy = touch.pos[1] - self.center_y
            self.center_x += dx * 0.1
            self.center_y += dy * 0.1
            return True
        return super().on_touch_move(touch)
    
    def _create_ripple(self, pos):
        """Create ripple effect at touch position"""
        # This would create expanding ripple animation
        # For now, just trigger a pulse
        self.pulse_intensity = 1.0
        
        # Log interaction for clinical tracking
        self.logger.info(f"Bubble touched at position: {pos}")
    
    def reset_state(self):
        """Reset bubble to default state"""
        self.bubble_radius = 100.0
        self.bubble_color = [0.5, 0.8, 1.0, 0.8]
        self.surface_roughness = 0.5
        self.metallic_shine = 0.3
        self.spikiness = 0.0
        self.pulse_intensity = 0.0
        self.center_x = self.width / 2
        self.center_y = self.height / 2

# Factory function for creating foam widget
def create_foam_widget(**kwargs) -> FoamWidget:
    """
    Factory function to create foam widget with default settings
    
    Returns:
        Configured FoamWidget instance
    """
    return FoamWidget(**kwargs)
