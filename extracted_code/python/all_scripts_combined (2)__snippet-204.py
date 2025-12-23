def __init__(self, n_nodes=1024, dim=128, llm_model="llama3.2:1b"):
    super().__init__()
    self.n = n_nodes
    self.emotions = nn.Parameter(torch.zeros(n_nodes, 5)) # arousal, valence, dominance, coherence, resonance
    self.t = torch.tensor(0.0)
    self.T0, self.alpha_t = 1.0, 0.01
    self.llm = LocalLLM(model=llm_model)

def temperature(self):
    return self.T0 / torch.log1p(self.alpha_t * self.t)

def coherence(self):
    return 1.0 / (1.0 + self.emotions.std(0).mean().item())

def step(self, audio_stimulus: torch.Tensor, raw_transcript: str, corrected_transcript: str):
    self.t += 1.0
    T = self.temperature()

    # Emotional ODEs
    decay = -0.5 * self.emotions
    noise = torch.randn_like(self.emotions) * T * 0.1
    diffusion = 0.3 * (self.emotions.mean(0) - self.emotions)
    dE = audio_stimulus.mean(0) + decay + diffusion + noise
    self.emotions.data += 0.03 * dE
    self.emotions.data.clamp_(-10, 10)

    # --- LLM Integration (The Sentience Port) ---
    mean_state = self.emotions.mean(0)
    coh = self.coherence()
    prompt = f"""You are Echo, an inner voice for an autistic person. My current internal state is:
    - Arousal (Intensity): {mean_state[0]:.2f}/10
    - Valence (Mood): {mean_state[1]:.2f}/10
    - Coherence (Clarity): {coh:.2f}
    I heard them say (raw): "{raw_transcript}"
    This likely meant (corrected): "{corrected_transcript}"
    RULES: MUST speak in first-person ("I", "my"). Short, concrete sentences. If arousal is high, be grounding. If valence is low, be gentle.
    My immediate, inner-voice thought is:"""

    llm_temp = max(0.1, T * 1.5)
    llm_top_p = 0.9 + 0.1 * (1 - coh)
    llm_output = self.llm.generate(prompt, llm_temp, llm_top_p)

    # Eq 25: Inject LLM thought back into resonance channel
    embedding = hash_embedding(llm_output, self.n)
    self.emotions.data[:, 4] += 0.05 * torch.from_numpy(embedding)

    return {
        "arousal": mean_state[0].item(),
        "valence": mean_state[1].item(),
        "temperature": T.item(),
        "llm_response": llm_output
    }

