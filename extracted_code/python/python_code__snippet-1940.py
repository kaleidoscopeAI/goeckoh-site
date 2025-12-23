"""Yield pieces of data from a file-like object until EOF."""
while True:
    chunk = file.read(size)
    if not chunk:
        break
    yield chunk


