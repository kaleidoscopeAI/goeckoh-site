def __init__(self, input_dim=8, num_heads=2, hidden_dim=32):  # Reduced for CPU
    super().__init__()
    self.self_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, input_dim)

def forward(self, x):
    attn_output, _ = self.self_attn(x, x, x)
    x = x + attn_output
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

