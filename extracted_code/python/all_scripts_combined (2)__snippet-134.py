class CrystallineHeart(nn.Module):
    """
    1024-node emotional lattice governed by ODEs
    
    dE/dt = drive + decay + diffusion + noise
    
    where:
    - drive = external_stimulus (voice arousal)
    - decay = -β * E
    - diffusion = γ * (global_mean - E)
    - noise = N(0,1) * T(t) * scale
    - T(t) = 1 / log(1 + k*t)  [annealing schedule]
    """
    
    def __init__(self, cfg: EchoConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Emotional state: [num_nodes, num_channels]
        self.emotions = nn.Parameter(
            torch.zeros(cfg.num_nodes, cfg.num_channels, device=self.device),
            requires_grad=False,
        )
        
        # Time counter for annealing
        self.register_buffer("t", torch.zeros(1, device=self.device))
        
        # LLM for sentience
        self.llm = LocalLLM(cfg)
        
    def reset(self):
        """Reset emotional state and time"""
        self.emotions.data.zero_()
        self.t.zero_()
    
    def temperature(self) -> float:
        """T(t) = 1 / log(1 + k*t) - logarithmic cooling"""
        t_val = float(self.t.item()) + 1.0
        k = self.cfg.anneal_k
        return float(1.0 / max(math.log(1.0 + k * t_val), 1e-6))
    
    def coherence(self) -> float:
        """Measure how aligned nodes are (0=scattered, 1=unified)"""
        E = self.emotions
        std_over_nodes = torch.std(E, dim=0)
        mean_std = float(torch.mean(std_over_nodes).item())
        return float(1.0 / (1.0 + mean_std))
    
    @torch.no_grad()
    def step(self, full_audio: np.ndarray, transcript: str) -> Dict[str, Any]:
        """
        Complete emotional + LLM update for one utterance
        
        Returns:
            {
                "arousal_raw": float,
                "T": float,
                "coherence": float,
                "emotions": torch.Tensor,
                "llm_output": str or None
            }
        """
        # ---- 1. Time & Temperature ----
        self.t += 1.0
        T_val = self.temperature()
        
        # ---- 2. Arousal Extraction ----
        full_audio = np.asarray(full_audio, dtype=np.float32)
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=-1)
        
        energy = float(np.sqrt(np.mean(full_audio**2) + 1e-12))
        arousal_raw = float(np.clip(energy * self.cfg.arousal_gain, 0.0, self.cfg.max_arousal))
        
        # External stimulus: [arousal, 0, 0, 1, 0] broadcast to all nodes
        stim_vec = torch.tensor(
            [arousal_raw, 0.0, 0.0, 1.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        external_stimulus = stim_vec.unsqueeze(0).repeat(self.cfg.num_nodes, 1)
        
        # ---- 3. ODE Update ----
        E = self.emotions
        
        drive = external_stimulus
        decay = -self.cfg.beta_decay * E
        global_mean = torch.mean(E, dim=0, keepdim=True)
        diffusion = self.cfg.gamma_diffusion * (global_mean - E)
        noise = torch.randn_like(E) * (T_val * self.cfg.noise_scale)
        
        dE_dt = drive + decay + diffusion + noise
        E.add_(self.cfg.dt * dE_dt)
        E.clamp_(-self.cfg.max_emotion, self.cfg.max_emotion)
        
        # ---- 4. LLM Integration (Equation 25) ----
        llm_output = None
        
        if transcript.strip():
            coh = self.coherence()
            mean_state = torch.mean(E, dim=0)
            mean_state_np = mean_state.cpu().numpy()
            
            arousal_mean = float(mean_state_np[0])
            valence_mean = float(mean_state_np[1])
            
            prompt = self._build_prompt(
                transcript=transcript,
                arousal=arousal_mean,
                valence=valence_mean,
                T_val=T_val,
                coherence=coh,
            )
            
            llm_temp = max(0.1, T_val * self.cfg.llm_temperature_scale)
            llm_top_p = self.cfg.llm_top_p_base + self.cfg.llm_top_p_spread * (1.0 - coh)
            
            llm_output = self.llm.generate(
                prompt=prompt,
                temperature=llm_temp,
                top_p=llm_top_p,
            )
            
            # Embed LLM output into resonance channel
            emb = self.llm.embed(llm_output, dim=self.cfg.embedding_dim)
            emb_t = torch.from_numpy(emb).to(self.device, dtype=torch.float32)
            
            if self.cfg.num_nodes <= self.cfg.embedding_dim:
                proj = emb_t[:self.cfg.num_nodes]
            else:
                reps = math.ceil(self.cfg.num_nodes / self.cfg.embedding_dim)
                tiled = emb_t.repeat(reps)
                proj = tiled[:self.cfg.num_nodes]
            
            proj = proj.view(self.cfg.num_nodes, 1)
            ch = self.cfg.embedding_channel
            
            if 0 <= ch < self.cfg.num_channels:
                E[:, ch:ch+1].add_(self.cfg.embedding_gain * proj)
                E.clamp_(-self.cfg.max_emotion, self.cfg.max_emotion)
        
        return {
            "arousal_raw": arousal_raw,
            "T": T_val,
            "coherence": self.coherence(),
            "emotions": self.emotions.detach().clone(),
            "llm_output": llm_output,
        }
    
    def _build_prompt(self, transcript: str, arousal: float, valence: float, 
                     T_val: float, coherence: float) -> str:
        """Build the inner voice prompt for DeepSeek"""
        return f"""You are my inner voice. I am {self.cfg.child_name}, an autistic child.

