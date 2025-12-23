def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 1):

    super().__init__()

    if not torch or not GCNConv:

        logging.error("PyTorch or PyTorch Geometric missing, GNN disabled")

        self.is_enabled = False

        return

    self.is_enabled = True

    self.conv1 = GCNConv(input_dim, hidden_dim)

    self.conv2 = GCNConv(hidden_dim, output_dim)


def forward(self, x, edge_index):

    if not self.is_enabled:

        return None

    try:

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        return x

    except Exception as e:

        logging.error(f"GNN forward failed: {e}")

        return None


