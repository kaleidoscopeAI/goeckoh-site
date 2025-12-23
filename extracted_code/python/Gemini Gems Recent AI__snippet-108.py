def __init__(self, input_dim: int = 7, hidden_dim: int = 16, latent_dim: int = 4):

    super().__init__()

    if not torch:

        logging.error("PyTorch missing, VAE disabled")

        self.is_enabled = False

        return

    self.is_enabled = True

    self.encoder = nn.Sequential(

        nn.Linear(input_dim, hidden_dim),

        nn.ReLU(),

        nn.Linear(hidden_dim, latent_dim * 2)

    )

    self.decoder = nn.Sequential(

        nn.Linear(latent_dim, hidden_dim),

        nn.ReLU(),

        nn.Linear(hidden_dim, input_dim),

        nn.Sigmoid()

    )


def reparameterize(self, mu, log_var):

    if not self.is_enabled:

        return None

    std = torch.exp(0.5 * log_var)

    eps = torch.randn_like(std)

    return mu + eps * std


def forward(self, x):

    if not self.is_enabled:

        return None, None, None

    try:

        h = self.encoder(x)

        mu, log_var = h.chunk(2, dim=-1)

        z = self.reparameterize(mu, log_var)

        return self.decoder(z), mu, log_var

    except Exception as e:

        logging.error(f"VAE forward failed: {e}")

        return None, None, None


