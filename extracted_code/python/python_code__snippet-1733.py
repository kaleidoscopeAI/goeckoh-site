"""
Memory-efficient FileCache: body is stored in a separate file, reducing
peak memory usage.
"""

def get_body(self, key: str) -> IO[bytes] | None:
    name = self._fn(key) + ".body"
    try:
        return open(name, "rb")
    except FileNotFoundError:
        return None

def set_body(self, key: str, body: bytes) -> None:
    name = self._fn(key) + ".body"
    self._write(name, body)

def delete(self, key: str) -> None:
    self._delete(key, "")
    self._delete(key, ".body")


