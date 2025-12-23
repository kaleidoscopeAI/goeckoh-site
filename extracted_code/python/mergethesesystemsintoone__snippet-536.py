def __init__(self, input_dim: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Tanh(),
    )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

