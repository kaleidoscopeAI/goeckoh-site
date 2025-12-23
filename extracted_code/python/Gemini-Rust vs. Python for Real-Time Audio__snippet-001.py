use nalgebra::{SVector, SMatrix};
use std::f32::consts::PI;

// --- CONFIGURATION ---
// The lattice dimensions: 10 Layers (Depth) x 24 Nodes (Width)
const ROWS: usize = 10;
const COLS: usize = 24;
const N_NODES: usize = ROWS * COLS;

// Physics Constants
const DAMPING: f32 = 0.98;         // Energy loss over time (prevents explosion)
const SPRING_K: f32 = 0.5;         // Stiffness of connections
const NEIGHBOR_COUPLING: f32 = 0.2;// How strongly nodes pull on each other
const MASS: f32 = 1.0;

/// The Crystalline Lattice
/// Instead of a neural net, we simulate a physical grid of springs and masses.
/// 
/// State Vector:
/// - positions: The displacement of each node from neutral.
/// - velocities: The momentum of each node.
pub struct CrystallineLattice {
    // We use a flat vector for memory locality (cache friendliness).
    // Accessing node (r, c) maps to index [r * COLS + c].
    positions: SVector<f32, N_NODES>,
    velocities: SVector<f32, N_NODES>,
    
    // Cached gradients for the 4-step Runge-Kutta solver (if needed),
    // or simple accumulators for Symplectic Euler.
    forces: SVector<f32, N_NODES>,
}

impl CrystallineLattice {
    pub fn new() -> Self {
        Self {
            positions: SVector::zeros(),
            velocities: SVector::zeros(),
            forces: SVector::zeros(),
        }
    }

    /// The "Heartbeat" of the system.
    /// Updates the physics simulation by one time step (dt).
    /// Uses Semi-Implicit Euler integration for stability and speed.
    pub fn update(&mut self, dt: f32) {
        // 1. Calculate Forces (The "Hamiltonian" Gradient)
        self.calculate_forces();

        // 2. Update Velocities (F = ma, so a = F/m)
        // Note: nalgebra optimizes these vector ops to SIMD automatically.
        self.velocities += self.forces * (dt / MASS);
        
        // 3. Apply Damping (Energy dissipation)
        self.velocities *= DAMPING;

        // 4. Update Positions
        self.positions += self.velocities * dt;
    }

    /// Calculates the net force on every node in the lattice.
    /// F_net = -k*x (Self) + coupling_forces (Neighbors)
    fn calculate_forces(&mut self) {
        // Reset forces
        self.forces.fill(0.0);

        // 1. Internal Restoring Force (Hooke's Law)
        // Nodes want to return to zero.
        self.forces -= self.positions * SPRING_K;

        // 2. Neighbor Coupling (The "Crystalline" Structure)
        // We iterate through the grid to calculate neighbor interactions.
        // This is the "O(N)" loop that Python struggles with.
        for r in 0..ROWS {
            for c in 0..COLS {
                let idx = r * COLS + c;
                let my_pos = self.positions[idx];

                // Check Right Neighbor
                if c + 1 < COLS {
                    let right_idx = r * COLS + (c + 1);
                    let diff = self.positions[right_idx] - my_pos;
                    let force = diff * NEIGHBOR_COUPLING;
                    
                    self.forces[idx] += force;
                    self.forces[right_idx] -= force; // Newton's 3rd Law
                }

                // Check Bottom Neighbor
                if r + 1 < ROWS {
                    let bottom_idx = (r + 1) * COLS + c;
                    let diff = self.positions[bottom_idx] - my_pos;
                    let force = diff * NEIGHBOR_COUPLING;

                    self.forces[idx] += force;
                    self.forces[bottom_idx] -= force;
                }
            }
        }
    }

    /// Injects audio energy ("Stimulus") into the system.
    /// This is how the user's voice shakes the lattice.
    pub fn inject_energy(&mut self, energy_spectrum: &[f32]) {
        // Map the frequency spectrum to the first row of the lattice.
        // If the spectrum is larger/smaller, we clamp or stride.
        let len = std::cmp::min(energy_spectrum.len(), COLS);
        
        for i in 0..len {
            // We perturb the velocity of the top layer nodes.
            // This creates a "wave" that propagates down through the layers.
            self.velocities[i] += energy_spectrum[i] * 5.0; 
        }
    }

    /// Reads the current "Emotional State" from the physics.
    /// - Arousal = Total Kinetic Energy (Temperature)
    /// - Valence = Symmetry/Harmonic alignment (Simplified here as stability)
    pub fn measure_affective_state(&self) -> (f32, f32, f32) {
        // Total Energy ~ Arousal
        let kinetic_energy: f32 = self.velocities.map(|v| v * v).sum();
        let arousal = (kinetic_energy * 0.1).clamp(0.0, 1.0);

        // Displacement Variance ~ Valence (Inverse)
        // High chaotic displacement = Negative Valence (Stress)
        // Low, ordered displacement = Positive Valence (Calm)
        let potential_energy: f32 = self.positions.map(|p| p * p).sum();
        let stress = (potential_energy * 0.05).clamp(0.0, 1.0);
        let valence = 1.0 - stress;

        // Coherence (Synchronization between layers)
        // Simple metric: variance between top and bottom layer average motion
        let coherence = 1.0 - (arousal - stress).abs().clamp(0.0, 1.0);

        (valence, arousal, coherence)
    }
}
