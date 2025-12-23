node_id: str = str(uuid.uuid4())
node_name: str = f"QSIN-Node-{uuid.uuid4().hex[:8]}"
server_url: str = "ws://localhost:8765"
server_mode: bool = False
port: int = 8765
host: str = "0.0.0.0"
dimension: int = 64
initial_energy: float = 20.0
replication_threshold: float = 100.0
discovery_enabled: bool = True

@classmethod
def from_dict(cls, data):
    return cls(**data)

