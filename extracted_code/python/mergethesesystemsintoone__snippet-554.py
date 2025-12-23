def __init__(self, input_dim: int) -> None:
    super().__init__()
    self.net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Tanh(),
    )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

