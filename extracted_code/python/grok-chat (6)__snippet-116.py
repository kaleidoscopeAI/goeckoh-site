class FinalMergedSystem:
    def __init__(self):
        self.heart = CrystallineHeart()
        self.metrics = AutismMetrics()

    def run(self):
        while True:
            input_text = input("Speak: ").strip()
            if not input_text:
                continue

            echoed, stimulus = auditory_motor_core(input_text, self.metrics)

            gcl, energy, disorder = self.heart.step(stimulus, input_text)

            core = DeepReasoningCore()
            response = core.execute(echoed, gcl)

            print(f"[GCL: {gcl:.2f}] [Energy: {energy:.2f}] [Disorder Entropy: {disorder:.2f}]")
            print(f"[Metrics: Attempts {self.metrics.attempts}, Success {self.metrics.success_rate():.2f}, Streak {self.metrics.streak}]")
            print(f"[Response]: {response}")
            time.sleep(1)

