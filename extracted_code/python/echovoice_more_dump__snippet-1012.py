"""Message for inter-node communication"""
message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
message_type: str = "ping"
sender_id: str = ""
receiver_id: str = ""  # Empty for broadcast
timestamp: float = field(default_factory=time.time)
content: Dict[str, Any] = field(default_factory=dict)

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for transmission"""
    return asdict(self)

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'NetworkMessage':
    """Create from dictionary"""
    return cls(**data)

