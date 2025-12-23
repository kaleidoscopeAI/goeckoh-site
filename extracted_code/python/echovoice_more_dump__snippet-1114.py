def __init__(self, vocab_size=256, d_model=128, nhead=4, num_layers=2):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.head = nn.Linear(d_model, vocab_size)
    self.d_model = d_model

def forward(self, x):
    # x: LongTensor (batch, seq)
    e = self.embed(x) * math.sqrt(self.d_model)
    out = self.encoder(e)
    logits = self.head(out)
    return logits

