def __init__(self, input_dim: int, n_actions: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions),
    )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(self.net(x), dim=-1)

