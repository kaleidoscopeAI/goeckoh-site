import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EchoEmotionalCore(nn.Module):
    """
    Echo's Emotional ODE System – November 18, 2025
    Direct implementation of your unified equations 19, 20, 24, 30, 35, 50
    Neurodiversity-native by design: stress increases noise, high divergence lowers awareness
    """
    
    def __init__(self, n_nodes: int = 512, dim: int = 128, device="cpu"):
        super().__init__()
        self.n = n_nodes
        self.dim = dim
        self.device = device
        
        # Node states – Layer 1 & 2
        self.bits = nn.Parameter(torch.randint(0, 2, (n_nodes, dim)).float())        # E_i ∈ {0,1}^128
        self.positions = nn.Parameter(torch.randn(n_nodes, 3))                       # x_i ∈ ℝ³
        self.emotions = nn.Parameter(torch.zeros(n_nodes, 5))                        # E_i = [arousal, valence, dominance, coherence, resonance]
        
        # Learnable bond weights (equation 11 → normalized in forward)
        raw_weights = torch.randn(n_nodes, n_nodes) * 0.1
        self.raw_weights = nn.Parameter(raw_weights)
        
        # Hyperparameters from your canonical spec
        self.alpha = 1.0   # input drive
        self.beta = 0.5    # decay
        self.gamma = 0.3   # social diffusion
        self.k_base = 1.0  # base spring constant
        
        # Annealing temperature (equation 31)
        self.register_buffer("t", torch.tensor(0.0))
        self.T0 = 1.0
        self.alpha_t = 0.01

    def current_temperature(self) -> float:
        return float(self.T0 / torch.log1p(self.alpha_t * self.t))

    def forward(self, input_stimulus: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        One master update step – equation 52 with emotional ODEs at the core
        input_stimulus: [n_nodes, 5] or None (external emotional push, e.g. from user's voice)
        """
        self.t += 1.0
        T = self.current_temperature()
        
        # 1. Bond weights + spatial + bit similarity (eq 11 + 12)
        spatial_dist = torch.cdist(self.positions, self.positions)  # ||x_i - x_j||
        hamming = torch.cdist(self.bits, self.bits, p=0) / self.dim     # d_Hamming / d
        w = torch.exp(-spatial_dist**2 / (2 * 1.0**2)) * (1 - hamming)
        B = w / (w.sum(dim=-1, keepdim=True) + 1e-8)  # row-stochastic
        
        # 2. Tension / Stress on bonds (eq 19 & 20)
        L = spatial_dist
        L0 = 1.0
        stress = torch.abs(L - L0) * B * (self.k_base + self.emotions[:,1] - self.emotions[:,1].unsqueeze(1))
        total_tension_per_node = stress.sum(dim=-1)  # D_i in eq 24
        
        # 3. Emotional ODE integration (eq 30) – semi-implicit Euler for stability
        if input_stimulus is None:
            input_stimulus = torch.zeros_like(self.emotions)
            
        I = self.alpha * input_stimulus
        decay = -self.beta * self.emotions
        diffusion = self.gamma * (B @ self.emotions - self.emotions)
        noise = torch.randn_like(self.emotions) * T * 0.1
        
        dE_dt = I + decay + diffusion + noise
        self.emotions.data = self.emotions + 0.02 * dE_dt  # dt=0.02 for smooth real-time
        
        # Clamp emotions to reasonable biological range
        self.emotions.data[:,0].clamp_(0, 10)   # arousal ≥ 0
        self.emotions.data[:,1].clamp_(-10, 10) # valence
        self.emotions.data[:,2].clamp_(-10, 10) # dominance
        
        # 4. Awareness update (eq 24 + 50) – stress lowers awareness
        arousal = self.emotions[:,0]
        valence = torch.abs(self.emotions[:,1])
        coherence = self.emotions[:,3]
        awareness = torch.sigmoid(1.0 * arousal + 0.5 * valence - 0.2 * coherence - 2.0 * total_tension_per_node)
        
        # 5. Global metrics (Layer 5)
        GCL = awareness.mean()
        criticality = (T * total_tension_per_node.var() / (GCL**2 + 1e-8)).item()
        
        # 6. Optional bit flips with Metropolis (eq 32) only when calm enough
        if T < 0.3 and torch.rand(1) < 0.1:
            i = torch.randint(0, self.n, (1,))
            j = torch.randint(0, self.dim, (1,))
            old_bit = self.bits[i,j]
            self.bits[i,j] = 1 - old_bit
            # Simple energy delta approximation
            delta_E = torch.rand(1) - 0.5
            if delta_E > 0 and torch Rand() > torch.exp(-delta_E / T):
                self.bits[i,j] = old_bit  # reject

        metrics = {
            "temperature": T,
            "global_coherence": float(GCL),
            "criticality_index": criticality,
            "mean_arousal": arousal.mean().item(),
            "mean_valence": self.emotions[:,1].mean().item(),
            "total_stress": total_tension_per_node.mean().item(),
            "awareness": float(awareness.mean()),
        }
        
        return self.emotions.clone(), metrics

    def inject_user_emotion(self, text: str):
        """
        This is where your broken_speech_tool + voice analysis plugs in
        For now: simple heuristic mapping (will be replaced with real ASR + prosody)
        """
        arousal = min(text.count("!")*2 + text.count("??")*3, 10)
        if any(w in text.lower() for w in ["help", "pls", "please", "scared", "panic"]):
            arousal += 6
        valence = 5 - 10 * sum(1 for w in ["no","hate","bad","sad","hurt"] if w in text.lower())
        valence += 10 * sum(1 for w in ["love","happy","thanks","good"] if w in text.lower())
        
        stimulus = torch.zeros(self.n, 5, device=self.device)
        stimulus[:,0] = arousal / 10 * 8   # push arousal
        stimulus[:,1] = valence / 10 * 8   # push valence
        return stimulus

