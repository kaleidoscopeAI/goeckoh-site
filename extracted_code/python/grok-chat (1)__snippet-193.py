def __init__(self):
    self.heart = CrystallineHeart()
    self.metrics = AutismMetrics()
    self.hid = HIDController()

def run(self):
    while True:
        input_text = input("Speak: ").strip()
        if not input_text:
            continue

        echoed, stimulus = auditory_motor_core(input_text, self.metrics, self.hid)

        gcl, energy, disorder = self.heart.step(stimulus, input_text)

        visualize_nodes(self.heart.nodes)  # Integrated visual

        core = DeepReasoningCore()
        response = core.execute(echoed, gcl)

        print(f"[GCL: {gcl:.2f}] [Energy: {energy:.2f}] [Disorder: {disorder:.2f}]")
        print(f"[Metrics: Attempts {self.metrics.attempts}, Rate {self.metrics.success_rate():.2f}, Streak {self.metrics.streak}]")
        print(f"[Response]: {response}")
        time.sleep(1)

