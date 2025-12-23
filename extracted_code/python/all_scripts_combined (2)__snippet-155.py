"""Consumes audio chunks from AudioIO and puts them into an asyncio Queue."""
for chunk in audio_io.chunk_generator():
    await q.put(chunk)
