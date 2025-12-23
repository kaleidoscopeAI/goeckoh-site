def __init__(self, agi):
    self.agi = agi

async def constant_hearing(self):
    while self.agi.running:
        audio_chunk = np.random.normal(0, 0.1, 512)
        features = self.agi.experience_processor.process_sound(audio_chunk)
        experience = {"type": "audio", "features": features, "energy_cost": 0.01}
        await self.agi.experience_queue.put(experience)
        await asyncio.sleep(0.05)

