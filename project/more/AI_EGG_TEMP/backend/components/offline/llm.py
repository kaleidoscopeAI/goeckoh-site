# backend/components/offline/llm.py
import torch
import torch.nn as nn
import math
from interfaces import BaseLLM

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        e = self.embed(x) * math.sqrt(self.d_model)
        out = self.encoder(e)
        logits = self.head(out)
        return logits

class OfflineLLM(BaseLLM):
    def __init__(self, device='cpu'):
        self.device = device
        self.model = MiniTransformer().to(self.device)
        torch.manual_seed(0)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model.eval()

    def _tokenize(self, text: str, max_len=128):
        b = text.encode('utf-8', errors='ignore')[:max_len]
        ids = [c for c in b]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    async def reflect(self, prompt: str) -> str:
        tok = self._tokenize(prompt)
        with torch.no_grad():
            logits = self.model(tok)
            avg = logits.mean(dim=1).squeeze(0)
            topk = torch.topk(avg, k=16).indices.cpu().numpy().tolist()
            chars = ''.join(chr((t % 94) + 32) for t in topk)
            reflection = f"Reflection: {chars}\nSummary: {prompt[:120]}"
            return reflection
