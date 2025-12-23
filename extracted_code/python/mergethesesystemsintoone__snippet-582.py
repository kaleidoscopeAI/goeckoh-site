data_type: str  # e.g., "text", "image", "numerical", "audio", "video"
data: Any
metadata: Dict[str, Any] = field(default_factory=dict)

def get_data(self) -> Any:
    return self.data

def get_metadata(self, key: str, default: Any = None) -> Any:
    return self.metadata.get(key, default)

def set_metadata(self, key: str, value: Any):
    self.metadata[key] = value

