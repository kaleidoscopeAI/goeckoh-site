def get(self, key: str) -> bytes | None:
    raise NotImplementedError()

def set(
    self, key: str, value: bytes, expires: int | datetime | None = None
) -> None:
    raise NotImplementedError()

def delete(self, key: str) -> None:
    raise NotImplementedError()

def close(self) -> None:
    pass


