def __init__(self, input_dim=8, num_heads=4, hidden_dim=64):
    super().__init__()
    self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, input_dim)

def forward(self, x):
    attn_output, _ = self.attn(x, x, x)
    x = x + attn_output
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

