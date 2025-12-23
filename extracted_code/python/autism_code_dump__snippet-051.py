"""
Implements the Crystalline Heart, a 1024-node ODE lattice that provides
a continuous, time-evolving model of the system's internal affective state.
"""
def __init__(self, config: HeartConfig = HEART_CONFIG) -> None:
    self.config = config
    self.rng = np.random.default_rng(seed=42)
    # Initialize the lattice state: [nodes, channels]
    self.lattice_state = self.rng.uniform(
        -0.1, 0.1, (config.n_nodes, config.n_channels)
    ).astype(np.float32)
    self.temperature: float = 1.0 # Annealing temperature

def _stimulus_from_event(self, event: EchoEvent) -> np.ndarray:
    """Create a stimulus vector from an EchoEvent."""
    # Simple mapping for now: audio energy affects 'arousal' and 'energy' channels
    stimulus = np.zeros(self.config.n_channels, dtype=np.float32)

    # Channel 0: arousal, Channel 4: energy
    # Let's assume event.meta['energy'] is a normalized audio energy
    audio_energy = event.meta.get("energy", 0.5)
    stimulus[0] = audio_energy * 0.5 
    stimulus[4] = audio_energy * 0.3

    # Let's say text length affects stress (channel 2)
    length_factor = np.clip(len(event.text_clean) / 100.0, 0, 2.0)
    stimulus[2] = length_factor * 0.1

    return stimulus

def _update_emotion_field(self, stimulus: np.ndarray) -> None:
    """
    Performs one vectorized Euler step of the ODE for the entire lattice.
    dE/dt = alpha*I - beta*E + gamma*(mean(E) - E) + noise
    """
    E = self.lattice_state

    # Calculate terms of the ODE
    drive = self.config.alpha * stimulus[np.newaxis, :]  # Apply stimulus to all nodes
    decay = -self.config.beta * E

    # Diffusion term (fully-connected field)
    mean_state = np.mean(E, axis=0, keepdims=True)
    diffusion = self.config.gamma * (mean_state - E)

    # Stochastic noise, scaled by temperature
    noise = self.rng.normal(
        0, 
        self.config.noise_level * self.temperature, 
        E.shape
    ).astype(np.float32)

    # Update the state
    dE = drive + decay + diffusion + noise
    self.lattice_state += self.config.dt * dE

    # Clip to prevent runaway values
    np.clip(
        self.lattice_state,
        -self.config.max_emotion_value,
        self.config.max_emotion_value,
        out=self.lattice_state
    )

def _anneal(self) -> None:
    """Slowly cool the system by reducing temperature."""
    # Simple decay for now; can be replaced with the 1/log(t) schedule later
    self.temperature = max(0.1, self.temperature * 0.995)

def _calculate_global_metrics(self) -> HeartMetrics:
    """Aggregate the 1024-node state into a single HeartMetrics snapshot."""
    E = self.lattice_state

    # Aggregate each channel
    # Assuming channel indices: 0:arousal, 1:valence, 2:stress, 3:harmony, 4:energy
    stress = float(np.mean(np.abs(E[:, 2])))
    harmony = 1.0 / (1.0 + float(np.mean(np.std(E, axis=0)))) # GCL-like metric
    energy = float(np.mean(E[:, 4]))

    # Confidence as inverse of overall variance
    confidence = 1.0 / (1.0 + float(np.var(E)))

    return HeartMetrics(
        timestamp=now_ts(),
        stress=np.clip(stress, 0, 1),
        harmony=np.clip(harmony, 0, 1),
        energy=np.clip(energy, 0, 2),
        confidence=np.clip(confidence, 0, 1),
        temperature=self.temperature
    )

def update_from_event(self, event: EchoEvent) -> HeartMetrics:
    """
    The main public method. Updates the heart state based on a new
    utterance and returns the new global metrics.
    """
    stimulus = self._stimulus_from_event(event)
    self._update_emotion_field(stimulus)
    self._anneal()
    return self._calculate_global_metrics()
    from __future__ import annotations

