class CrystallineHeart:
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

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from core.settings import HeartSettings


class EchoCrystallineHeart(nn.Module):
    """
    Emotional lattice + LLM integration.
    """

    def __init__(self, cfg: HeartSettings):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Emotions tensor: [num_nodes, num_channels]
        self.emotions = nn.Parameter(
            torch.zeros(cfg.num_nodes, cfg.num_channels, device=self.device),
            requires_grad=False,
        )
        # Time (discrete steps)
        self.register_buffer("t", torch.zeros(1, device=self.device))

    @torch.no_grad()
    def reset(self):
        self.emotions.zero_()
        self.t.zero_()

    @torch.no_grad()
    def temperature(self) -> float:
        """
        T(t) = 1 / log(1 + k t) (eq 31 style)
        """
        t_val = float(self.t.item()) + 1.0  # avoid log(0)
        k = self.cfg.anneal_k
        return float(1.0 / max(math.log(1.0 + k * t_val), 1e-6))

    @torch.no_grad()
    def coherence(self) -> float:
        """
        Simple coherence metric in [0,1]:
        - 1 = all nodes identical
        - 0 = highly scattered
        Implemented as:
        coherence = 1 / (1 + mean_std)
        """
        # [N, C]
        E = self.emotions
        # std over nodes, then mean over channels
        std_over_nodes = torch.std(E, dim=0)
        mean_std = float(torch.mean(std_over_nodes).item())
        return float(1.0 / (1.0 + mean_std))

    @torch.no_grad()
    def step(self, full_audio: np.ndarray) -> dict:
        """
        One full emotional update after a completed utterance.
        """
        # ---- 1. Update time + temperature --------------------------------
        self.t += 1.0
        T_val = self.temperature()
        # ---- 2. Extract arousal from waveform ----------------------------
        full_audio = np.asarray(full_audio, dtype=np.float32)
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=-1)
        # RMS energy
        energy = float(np.sqrt(np.mean(full_audio**2) + 1e-12))
        arousal_raw = float(np.clip(energy * self.cfg.arousal_gain, 0.0, self.cfg.max_arousal))
        # external stimulus vector: [arousal, 0, 0, 1, 0]
        stim_vec = torch.tensor(
            [arousal_raw, 0.0, 0.0, 1.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        # External stimulus broadcast to all nodes
        external_stimulus = stim_vec.unsqueeze(0).repeat(self.cfg.num_nodes, 1)
        # ---- 3. ODE update: dE/dt = drive + decay + diffusion + noise ----
        E = self.emotions  # [N, C]
        # drive term: α * I_i(t) (we let α ≈ 1 here)
        drive = external_stimulus
        # decay: -β * E
        decay = -self.cfg.beta_decay * E
        # diffusion: γ * (global_mean - E)
        global_mean = torch.mean(E, dim=0, keepdim=True)
        # [1, C]
        diffusion = self.cfg.gamma_diffusion * (global_mean - E)
        # noise: N(0, 1) * T * noise_scale
        noise = torch.randn_like(E) * (T_val * self.cfg.noise_scale)
        dE_dt = drive + decay + diffusion + noise
        # Euler integration: E(t+1) = E(t) + dt * dE/dt
        E.add_(self.cfg.dt * dE_dt)
        E.clamp_(-self.cfg.max_abs, self.cfg.max_abs)

        return {
            "arousal_raw": arousal_raw,
            "external_stimulus": external_stimulus.detach().clone(),
            "T": T_val,
            "coherence": self.coherence(),
            "emotions": self.emotions.detach().clone(),
        }
        #!/usr/bin/env python3
