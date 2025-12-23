"""Combine small chunks from sounddevice into fixed-length windows."""
buffer: Optional[np.ndarray] = None
chunk_frames = int(seconds * samplerate)
for block in stream:
    buffer = block if buffer is None else np.concatenate([buffer, block], axis=0)
    while buffer is not None and len(buffer) >= chunk_frames:
        yield buffer[:chunk_frames]
        buffer = buffer[chunk_frames:] if len(buffer) > chunk_frames else None
