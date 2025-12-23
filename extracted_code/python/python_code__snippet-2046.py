"""Information about a core metadata file associated with a distribution."""

hashes: Optional[Dict[str, str]]

def __post_init__(self) -> None:
    if self.hashes is not None:
        assert all(name in _SUPPORTED_HASHES for name in self.hashes)


