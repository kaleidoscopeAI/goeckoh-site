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
