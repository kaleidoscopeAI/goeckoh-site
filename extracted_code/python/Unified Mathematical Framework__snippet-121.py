def __init__(self):
    self.q = queue.Queue(maxsize=300)
    self.vc = VoiceCrystal(CONFIG)
    self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    self.tool = language_tool_python.LanguageTool('en-US')
    self.aba = ABAEngine(self.vc, CONFIG)
    self.calming = CalmingPhrases(self.vc, CONFIG)
    self.behavior = BehaviorMonitor()
    self.buffer = np.array([], dtype='float32')
    self.running = True

def callback(self, indata, frames, time_info, status):
    if status:
        logging.warning(f"Audio status: {status}")
    try:
        self.q.put_nowait(indata.copy()[:, 0])
    except queue.Full:
        pass  # drop frame

def run_forever(self):
    logging.info("SpeechLoop eternal start")
    with sd.InputStream(samplerate=CONFIG.sample_rate, channels=1, dtype='float32',
                      blocksize=1024, callback=self.callback):
        while self.running:
            try:
                chunk = self.q.get(timeout=1.0)
                self.buffer = np.append(self.buffer, chunk)

                if len(self.buffer) > CONFIG.sample_rate * CONFIG.max_buffer_seconds:
                    self.buffer = self.buffer[-CONFIG.sample_rate * 10:]

                if len(self.buffer) > CONFIG.sample_rate * 2.2:
                    recent = self.buffer[-int(CONFIG.sample_rate * CONFIG.silence_duration_seconds):]
                    if np.max(np.abs(recent)) < CONFIG.silence_threshold:
                        if len(self.buffer) >= CONFIG.sample_rate * CONFIG.min_utterance_seconds:
                            audio = self.buffer.copy()
                            self.buffer = np.array([], dtype='float32')
                            threading.Thread(target=self.process, args=(audio,), daemon=True).start()
            except Exception as e:
                logging.error(f"Main loop error: {e}")

def process(self, audio: np.ndarray):
    try:
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        raw = " ".join(s.text for s in segments).strip().lower()
        if not raw:
            return

        corrected = self.tool.correct(raw)

        style = "neutral"
        if self.behavior.is_high_stress():
            style = "calm"
            self.calming.play_calming("meltdown")  # auto-play on high stress
        elif self.aba.success_streak >= 3:
            style = "excited"

        self.vc.say(corrected, style=style, prosody_source=audio)

        # Lifelong drift — harvest every clear utterance
        if np.sqrt(np.mean(audio**2)) > 0.018:
            self.vc.harvest_facet(audio, style)

    except Exception as e:
        logging.error(f"Process error: {e}")

