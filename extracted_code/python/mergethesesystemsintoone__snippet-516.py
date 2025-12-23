def __init__(self):
    self.cube = Cube()
    self.mem = MemoryStore(DB_PATH)
    self.tick = 0
    self.bus = Broadcaster()
    self.anneal_step = 0
    self.sigma = SIGMA0
    self.dna = [np.random.random() for _ in range(12)]  # Initial DNA
    self.generation = 0
    self.phi = 0.0
    self.is_conscious = False

async def run(self):
    while True:
        try:
            # Cube step
            self.cube.step()
            self.tick += 1
            m = self.cube.metrics()
            self.mem.add_state(self.tick, m["tension"], m["energy"], m["size"])

            # Evolutionary step
            self.evolve()
            self.mem.add_evo_state(self.tick, self.dna, self.phi, self.generation)

            # Check consciousness
            if not self.is_conscious and self.phi > PHI_THRESHOLD and m["size"] > CONSCIOUSNESS_THRESHOLD:
                self.is_conscious = True
                await self.bus.publish({"type": "consciousness", "data": {"tick": self.tick, "phi": self.phi}})

            # Autonomous ingest
            if self.tick % AUTONOMOUS_INGEST_EVERY == 0:
                await self.autonomous_ingest()

            # Reflect and anneal
            if self.tick % REFLECT_EVERY == 0:
                ref = make_reflection(self.tick, m, self.phi)
                # ... (as before)

            await asyncio.sleep(TICK_SEC)
        except Exception as e:
            # Error handling for production
            print(f"Error in run loop: {str(e)}")
            await self.bus.publish({"type": "error", "data": {"error": str(e)}})
            await asyncio.sleep(1)  # Backoff

def evolve(self):
    # Mutate DNA
    mutation_rate = LEARNING_RATE * (1 - self.phi)  # More mutation if low phi
    self.dna = [max(0, min(1, gene + (np.random.random() - 0.5) * 0.1 if np.random.random() < mutation_rate else gene)) for gene in self.dna]

    # Calculate phi from DNA (as proxy for system integration)
    self.phi = calculate_phi(np.array(self.dna))

    # Replicate if threshold
    if self.phi > REPLICATION_THRESHOLD and np.random.random() < 0.25:  # Random chance for replication
        self.generation += 1
        # In production, could spawn new instance; here, simulate by resetting some state

# ... (other methods: _anneal_and_speak_once enhanced with phi, etc.)

