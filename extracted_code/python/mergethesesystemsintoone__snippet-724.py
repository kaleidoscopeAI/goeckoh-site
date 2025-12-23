def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
    self.fc = nn.Linear(hidden_size, hidden_size)
    self.classifier = nn.Linear(hidden_size, 2)  # Binary classification

def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lstm_out, _ = self.lstm(x)
    attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
    features = self.fc(attn_out)
    predictions = self.classifier(features)
    return predictions, attn_weights

