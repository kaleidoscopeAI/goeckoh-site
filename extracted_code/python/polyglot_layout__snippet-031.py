class GNNOracle(torch.nn.Module):

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

class RLPolicy(torch.nn.Module):

  def __init__(self, input_dim: int, n_actions: int) -> None:

    super().__init__()

    self.net = torch.nn.Sequential(

        torch.nn.Linear(input_dim, 128),

        torch.nn.ReLU(),

        torch.nn.Linear(128, 128),

        torch.nn.ReLU(),

        torch.nn.Linear(128, n_actions),

    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    return torch.nn.functional.softmax(self.net(x), dim=-1)

# KnowledgeProcessor

class KnowledgeProcessor:

  def __init__(self) -> None:

    self.processed_hashes: set[str] = set()

    self.model = SentenceTransformer('all-MiniLM-L6-v2')

  def process_web_content(self, content: Dict[str, str]) -> Optional[List[Dict[str, str]]]:

    content_hash = hash(json.dumps(content))

    if content_hash in self.processed_hashes:

        return None

    concepts: List[Dict[str, str]] = []

    base_meta = {

        "source": content.get("url", ""),

        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),

