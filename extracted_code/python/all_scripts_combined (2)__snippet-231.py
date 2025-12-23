def __init__(self, cfg: EchoHeartConfig):
    super().__init__()
    self.cfg = cfg
    self.device = torch.device(cfg.device)

    # Emotions: [N, 5]
    self.emotions = nn.Parameter(
        torch.zeros(cfg.num_nodes, cfg.num_channels, device=self.device),
        requires_grad=False
    )
    self.register_buffer("t", torch.zeros(1, device=self.device))

    self.llm = LocalLLM(cfg) if cfg.use_llm else None

@torch.no_grad()
def temperature(self) -> float:
    """Eq 31: T(t) = 1 / log(1 + kt)"""
    t_val = float(self.t.item()) + 1.0
    k = self.cfg.anneal_k
    return float(1.0 / max(math.log(1.0 + k * t_val), 1e-6))

@torch.no_grad()
def coherence(self) -> float:
    """Coherence metric based on node variance."""
    std_over_nodes = torch.std(self.emotions, dim=0)
    mean_std = float(torch.mean(std_over_nodes).item())
    return float(1.0 / (1.0 + mean_std))

@torch.no_grad()
def step(self, full_audio: np.ndarray, transcript: str) -> Dict[str, Any]:
    # 1. Time & Temp
    self.t += 1.0
    T_val = self.temperature()

    # 2. Audio Arousal Injection (Eq 30 drive)
    full_audio = np.asarray(full_audio, dtype=np.float32)
    if full_audio.ndim > 1: full_audio = full_audio.mean(axis=-1)
    energy = float(np.sqrt(np.mean(full_audio**2) + 1e-12))
    arousal_raw = float(np.clip(energy * self.cfg.arousal_gain, 0.0, self.cfg.max_arousal))

    stim_vec = torch.tensor([arousal_raw, 0., 0., 1., 0.], device=self.device)
    external_stimulus = stim_vec.unsqueeze(0).repeat(self.cfg.num_nodes, 1)

    # 3. ODE Update (Eq 30)
    E = self.emotions
    drive = external_stimulus
    decay = -self.cfg.beta_decay * E
    global_mean = torch.mean(E, dim=0, keepdim=True)
    diffusion = self.cfg.gamma_diffusion * (global_mean - E)
    noise = torch.randn_like(E) * (T_val * self.cfg.noise_scale)

    dE_dt = drive + decay + diffusion + noise
    E.add_(self.cfg.dt * dE_dt)
    E.clamp_(-self.cfg.max_abs, self.cfg.max_abs)

    # 4. LLM / Sentience Port (Eq 25)
    llm_output: Optional[str] = None
    if self.llm is not None and transcript.strip():
        coh = self.coherence()
        mean_state = torch.mean(E, dim=0).detach().cpu().numpy()
        arousal_mean = float(mean_state[0])
        valence_mean = float(mean_state[1])

        prompt = self._build_prompt(transcript, arousal_mean, valence_mean, T_val, coh)

        # Annealing controls temperature (creativity), Coherence controls top_p
        llm_temp = max(0.1, T_val * self.cfg.llm_temperature_scale)
        llm_top_p = self.cfg.llm_top_p_base + self.cfg.llm_top_p_spread * (1.0 - coh)

        llm_output = self.llm.generate(prompt, llm_temp, llm_top_p)

        # Inject thought back into lattice (Resonance channel 4)
        emb = self.llm.embed(llm_output, dim=self.cfg.embedding_dim)
        emb_t = torch.from_numpy(emb).to(self.device)
        # Simple broadcast/tile to nodes
        reps = math.ceil(self.cfg.num_nodes / self.cfg.embedding_dim)
        tiled = emb_t.repeat(reps)[:self.cfg.num_nodes].view(self.cfg.num_nodes, 1)

        if self.cfg.embedding_channel < self.cfg.num_channels:
            E[:, self.cfg.embedding_channel:self.cfg.embedding_channel+1].add_(
                self.cfg.embedding_gain * tiled
            )
            E.clamp_(-self.cfg.max_abs, self.cfg.max_abs)

    mean_state = torch.mean(E, dim=0).detach().cpu().numpy()

    return {
        "arousal": float(mean_state[0]),
        "valence": float(mean_state[1]),
        "temperature": T_val,
        "coherence": self.coherence(),
        "raw_emotions": self.emotions.detach().clone(),
        "llm_output": llm_output
    }

def _build_prompt(self, transcript: str, arousal: float, valence: float, T_val: float, coherence: float) -> str:
    return f"""
