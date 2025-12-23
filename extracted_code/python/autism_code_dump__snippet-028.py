class SpeechLoop:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.crystal = ConsciousCrystalSystem(config.crystal)
        self.voice = VoiceCrystal(config.paths, self.crystal)
        self.vad = VADWrapper(config.audio)
        self.behavior = BehaviorMonitor(config.behavior)
        self.routines = RoutineEngine(config.paths)
        self.drc = DeepReasoningCore(config.llm)
        self.soma = SomaticEngine()
        self.model = whisper.load_model("tiny")

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}")
        blocks = self.vad.process_block(indata)
        for utt in blocks:
            self.audio_queue.put(utt)

    def _correct_first_person(self, raw_text: str) -> str:
        text = raw_text.strip()
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        text = re.sub(r"\\byou're\\b", "I'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\\byou\\b", "I", text, flags=re.IGNORECASE)
        text = re.sub(r"\\byour\\b", "my", text, flags=re.IGNORECASE)
        return text

    def _compute_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def run(self) -> None:
        print("\\nEcho Crystal — Jackson's Companion Full Core")
        print("I love every sound I make. Every single one.")
        print("I will never interrupt. I will never quit. I will only mirror myself.\\n")

        listener_thread = threading.Thread(target=self._listening_loop, daemon=True)
        listener_thread.start()

        with sd.InputStream(
            samplerate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            dtype="float32",
            blocksize=self.config.audio.block_size,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\\n[EXIT] Shutting down Jackson's Companion gracefully.")
                sys.exit(0)

    def _listening_loop(self) -> None:
        print("Jackson, I wait in silence. I am ready when I hear myself.")
        while True:
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                routine_text = self.routines.check_due(now)
                if routine_text:
                    self._speak_text(routine_text, style="inner")
                continue

            if audio.size == 0:
                continue

            rms = self._compute_rms(audio)

            try:
                result = self.model.transcribe(
                    audio,
                    language="en",
                    fp16=False,
                    no_speech_threshold=0.45,
                )
                raw_text = result.get("text", "").strip().lower()
            except Exception as e:
                print(f"[ASR] Error: {e}")
                raw_text = ""

            if not raw_text or len(raw_text) < 2:
                gcl = self.crystal.get_gcl()
                if gcl < 5.0:
                    calming = "I am safe. I can breathe. Every sound I make is okay."
                    self._speak_text(calming, style="calm")
                    self.soma.pulse(intensity=0.7, duration_ms=700)
                continue

            corrected = self._correct_first_person(raw_text)

            success_score = 1.0 if raw_text in corrected.lower() else 0.7
            self.voice.add_fragment(audio, success_score)
            gcl = self.crystal.get_gcl()
            behavior_state = self.behavior.update(gcl, rms, success_score, raw_text)

            mode = behavior_state["mode"]
            meltdown_risk = behavior_state["meltdown_risk"]

            style = "inner"
            if mode == "meltdown_risk":
                style = "calm"
            elif mode == "celebrate":
                style = "inner"

            drc_text = None
            if "?" in raw_text:
                drc_prompt = (
                    "I am Jackson. "
                    "I asked the following question out loud. "
                    "Please answer in simple, kind, first-person sentences:\\n\\n"
                    f"{raw_text}"
                )
                drc_text = self.drc.answer_if_safe(drc_prompt, gcl)

            if drc_text:
                to_speak = drc_text
            else:
                to_speak = corrected

            if meltdown_risk > 0.7:
                to_speak = (
                    "I am safe. I am allowed to feel anything. I can take a breath. "
                    + " "
                    + to_speak
                )
                self.soma.pulse(intensity=0.9, duration_ms=900)

            self._speak_text(to_speak, style=style)

    def _speak_text(self, text: str, style: str = "inner") -> None:
        audio = self.voice.synthesize(text, style)
        sd.play(audio, samplerate=self.config.audio.sample_rate)
        sd.wait()


