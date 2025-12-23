def __init__(self):
    self.heart = CrystallineHeart()
    self.drc = DeepReasoningCore()

def run(self):
    while True:
        input_text = simulate_audio_input()
        if not input_text:
            continue

        echoed, arousal, latency = auditory_motor_core(input_text)

        gcl = self.heart.step(arousal, latency)

        response = self.drc.execute(echoed, gcl)

        print(f"[GCL: {gcl:.2f}] [Gated Response]: {response}")
        time.sleep(1)

