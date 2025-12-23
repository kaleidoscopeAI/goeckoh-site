def __init__(self, hybrid: Any, cognitive: Any, error_correction: Any, hardware: Any, config: Optional[Dict[str, Any]] = None) -> None:
    self.hybrid = hybrid
    self.cognitive = cognitive
    self.error_correction = error_correction
    self.hardware = hardware
    self.config = config or {}
    self.running = False

def step(self, dt: float = 0.01) -> None:
    self.hybrid.gradient_flow.step(dt)
    self.hybrid.metropolis.step()
    self.error_correction.detect_and_repair(self.hybrid.state)
    # cognitive update placeholder
    # invariants could be checked here

def run(self, dt: float = 0.01, steps: Optional[int] = None) -> None:
    self.running = True
    try:
        step_i = 0
        while self.running:
            self.step(dt)
            step_i += 1
            if steps is not None and step_i >= steps:
                break
            time.sleep(dt)
    finally:
        self.running = False

def stop(self) -> None:
    self.running = False

