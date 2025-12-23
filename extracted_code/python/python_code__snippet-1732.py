"""
Traditional FileCache: body is stored in memory, so not suitable for large
downloads.
"""

def delete(self, key: str) -> None:
    self._delete(key, "")


