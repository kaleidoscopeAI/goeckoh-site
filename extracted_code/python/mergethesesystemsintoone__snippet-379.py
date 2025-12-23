class EnergyState:
    magnitude: float
    direction: np.array
    frequency: float

    def propagate(self, distance: float) -> float:
        """Calculate energy propagation over distance."""
        return self.magnitude * np.exp(-distance / DECAY_CONSTANT)

class MemoryState:
    energy_level: float = 0.0
    activation: float = 0.0
    connections: Dict[int, float] = field(default_factory=dict)

    def update_state(self, energy_input: float, tension: float):
        """Update the memory state based on energy input and tension."""
        self.energy_level = energy_input
        self.activation = np.tanh((energy_input * tension) / ACTIVATION_THRESHOLD)

class MemoryPoint:
    id: int
    position: np.array
    energy: float = 0.0
    tension: float = 0.0
    memory_state: MemoryState = field(default_factory=MemoryState)

class StringNetwork:
    def __init__(self, cube_size: float = 10.0):
        self.cube_size = cube_size
        self.memory_points: Dict[int, MemoryPoint] = {}
        self.connections: List[Tuple[int, int]] = []
        self.initialize_cube()

    def initialize_cube(self):
        """Initialize memory points in a 3D cube."""
        for x, y, z in product(range(int(self.cube_size)), repeat=3):
            point_id = len(self.memory_points)
            self.memory_points[point_id] = MemoryPoint(
                id=point_id, position=np.array([x, y, z])
            )

    def calculate_string_tension(self, point1: MemoryPoint, point2: MemoryPoint) -> float:
        """Calculate tension between two points."""
        distance = np.linalg.norm(point1.position - point2.position)
        return np.exp(-distance / DECAY_CONSTANT)

    def update_field(self, source_point: MemoryPoint, energy_state: EnergyState):
        """Update energy field for all memory points."""
        for point in self.memory_points.values():
            distance = np.linalg.norm(point.position - source_point.position)
            energy_input = energy_state.propagate(distance)
            point.memory_state.update_state(energy_input, point.tension)

    def simulate_connections(self):
        """Create connections and update tensions dynamically."""
        for point_id, point in self.memory_points.items():
            neighbors = self.get_neighbors(point)
            for neighbor_id in neighbors:
                tension = self.calculate_string_tension(point, self.memory_points[neighbor_id])
                point.tension += tension
                self.connections.append((point_id, neighbor_id))

    def get_neighbors(self, point: MemoryPoint) -> List[int]:
        """Find neighboring points within a unit distance."""
        neighbors = []
        for other_id, other_point in self.memory_points.items():
            if np.linalg.norm(point.position - other_point.position) <= 1.0 and point.id != other_id:
                neighbors.append(other_id)
        return neighbors

    def calculate_system_tension(self) -> float:
        """Calculate total system tension as a sum of all connections."""
        total_tension = 0.0
        for point_id, point in self.memory_points.items():
            total_tension += point.tension
        return total_tension

    # Example usage
    network = StringNetwork(cube_size=5.0)
    energy_state = EnergyState(magnitude=5.0, direction=np.array([1, 0, 0]), frequency=1.0)
    source_point = network.memory_points[0]

    # Update the field and simulate connections
    network.update_field(source_point, energy_state)
    network.simulate_connections()

    # Log total system tension
    total_tension = network.calculate_system_tension()
    print(f"Total System Tension: {total_tension}")

import { useEffect, useState, useRef } from 'react';
import { create, all } from 'mathjs';



    const { width, height } = ctx.canvas;
    ctx.clearRect(0, 0, width, height);

    // Set up 3D projection matrix
    const perspective = 500;
    const scale = 40;

    // Project 3D point to 2D
    const project = (point) => {
      const [x, y, z] = point;
      const rotX = math.cos(rotation.x) * x - math.sin(rotation.x) * z;
      const rotZ = math.sin(rotation.x) * x + math.cos(rotation.x) * z;
      const rotY = math.cos(rotation.y) * y - math.sin(rotation.y) * rotZ;
      const projZ = math.sin(rotation.y) * y + math.cos(rotation.y) * rotZ;
      
      const projectScale = scale / (perspective - projZ);
      return [
        width/2 + rotX * projectScale,
        height/2 + rotY * projectScale,
        projZ
      ];
    };

    // Draw tension field
    if (state.tension_field) {
      const { X, Y, Z, tension } = state.tension_field;
      const maxTension = math.max(tension.flat());
      
      // Sample points for visualization
      const sampleRate = 5;
      for (let i = 0; i < X.length; i += sampleRate) {
        for (let j = 0; j < Y.length; j += sampleRate) {
          for (let k = 0; k < Z.length; k += sampleRate) {
            const point = [X[i], Y[j], Z[k]];
            const t = tension[i][j][k] / maxTension;
            const [px, py] = project(point);
            
            ctx.fillStyle = `rgba(255, 0, 0, ${t * 0.2})`;
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      }
    }

    // Draw connections
    ctx.strokeStyle = 'rgba(100, 100, 255, 0.5)';
    ctx.lineWidth = 1;
    
    for (const [i, j] of state.connections) {
      const point1 = state.points[i];
      const point2 = state.points[j];
      
      const [x1, y1] = project(point1[0]);
      const [x2, y2] = project(point2[0]);
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Draw memory points
    for (const [position, energy, activation] of state.points) {
      const [px, py, pz] = project(position);
      const radius = 5 + energy * 2;
      
      // Point glow based on activation
      const gradient = ctx.createRadialGradient(px, py, 0, px, py, radius * 2);
      gradient.addColorStop(0, `rgba(0, 255, 255, ${activation})`);
      gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Point core
      ctx.fillStyle = `rgb(0, ${Math.floor(energy * 255)}, 255)`;
      ctx.beginPath();
      ctx.arc(px, py, radius/2, 0, 2 * Math.PI);
      ctx.fill();
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Handle rotation with mouse
    const handleMouseMove = (e) => {
      if (e.buttons === 1) {
        setRotation(prev => ({
          x: prev.x + e.movementX * 0.01,
          y: prev.y + e.movementY * 0.01
        }));
      }
    };
    
    canvas.addEventListener('mousemove', handleMouseMove);
    
    // Animation loop
    let animationFrame;
    const animate = () => {
      if (cubeState) {
        drawCube(ctx, cubeState);
      }
      animationFrame = requestAnimationFrame(animate);
    };
    animate();
    
    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrame);
    };

    const updateInterval = setInterval(() => {
      // Simulate cube state changes
      const points = Array(10).fill(0).map(() => [
        [Math.random() * 10, Math.random() * 10, Math.random() * 10],
        Math.random(),
        Math.random()
      ]);
      
      const connections = [];
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          if (Math.random() < 0.3) {
            connections.push([i, j]);
          }
        }
      }
      
      setCubeState({
        points,
        connections,
        tension_field: {
          X: Array(10).fill(0),
          Y: Array(10).fill(0),
          Z: Array(10).fill(0),
          tension: Array(10).fill(Array(10).fill(Array(10).fill(Math.random())))
        }
      });
    }, 50);
    
    return () => clearInterval(updateInterval);

    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Cube Memory System Visualization</h2>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          className="border border-gray-300 rounded"
        />
        <div className="absolute top-2 left-2 text-sm text-gray-600">
          Drag to rotate
        </div>
      </div>
    </div>

import logging
from typing import List, Dict
from dash import Dash, dcc, html
import plotly.graph_objs as go
import numpy as np
from scipy.interpolate import CubicSpline
import random
import time

