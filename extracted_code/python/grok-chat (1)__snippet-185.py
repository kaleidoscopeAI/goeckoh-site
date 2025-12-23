def __init__(self):
    self.heart = CrystallineHeart()

def run(self):
    while True:
        input_text = input("Speak: ").strip()
        if not input_text:
            continue

        echoed, stimulus = auditory_motor_core(input_text)

        gcl, energy = self.heart.step(stimulus)

        core = DeepReasoningCore()
        response = core.execute(echoed, gcl)

        print(f"[GCL: {gcl:.2f}] [Energy: {energy:.2f}] [Response]: {response}")
        time.sleep(1)

