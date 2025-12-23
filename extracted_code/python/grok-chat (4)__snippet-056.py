def __init__(self):
    self.q = queue.Queue(maxsize=300)
    self.vc = VoiceCrystal(CONFIG)
    self.model = None
    self.tool = language_tool_python.LanguageTool('en-US')
    self.aba = ABAEngine(self.vc, CONFIG)
    self.buffer = np.array([], dtype='float32')
    self.running = True

    try:
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        logging.info("Whisper model loaded successfully")
    except Exception as e:
        logging.critical(f"Whisper failed to load: {e}")
        self.model = None

def callback(self, indata, frames, time_info, status):
    if status:
        logging.warning(f"Audio callback status: {status}")
    try:
        self.q.put_nowait(indata.copy()[:, 0])
    except queue.Full:
        logging.warning("Audio queue full — dropping frames")

def run_forever(self):
    logging.info("SpeechLoop started — listening forever")
    try:
        with sd.InputStream(samplerate=CONFIG.sample_rate, channels=1, dtype='float32',
                          blocksize=1024, callback=self.callback):
            while self.running:
                try:
                    chunk = self.q.get(timeout=1.0)
                    self.buffer = np.append(self.buffer, chunk)

                    if len(self.buffer) > CONFIG.sample_rate * CONFIG.max_buffer_seconds:
                        self.buffer = self.buffer[-CONFIG.sample_rate * 8:]

                    if len(self.buffer) > CONFIG.sample_rate * 2.0:
                        recent = self.buffer[-int(CONFIG.sample_rate * CONFIG.silence_duration_seconds):]
                        if np.max(np.abs(recent)) < CONFIG.silence_threshold:
                            if len(self.buffer) >= CONFIG.sample_rate * CONFIG.min_utterance_seconds:
                                audio = self.buffer.copy()
                                self.buffer = np.array([], dtype='float32')
                                threading.Thread(target=self.process_utterance, args=(audio,), daemon=True).start()
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Loop inner error: {e}\n{traceback.format_exc()}")
    except Exception as e:
        logging.critical(f"Audio stream failed: {e}\n{traceback.format_exc()}")
        logging.info("SpeechLoop stopped")

def process_utterance(self, audio: np.ndarray):
    try:
        if self.model is None:
            logging.error("Whisper not available — skipping utterance")
            return

        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True, beam_size=5)
        raw_text = " ".join(seg.text for seg in segments).strip().lower()
        if not raw_text:
            return

        corrected_text = self.tool.correct(raw_text)

        style = "neutral"
        if self.aba.current_emotion in ["anxious", "high_energy", "meltdown_risk"]:
            style = "calm"
        elif self.aba.success_streak >= 3:
            style = "excited"

        self.vc.say(corrected_text, style=style, prosody_source=audio)

        # Harvest if clear enough
        rms = np.sqrt(np.mean(audio**2).mean())
        if rms > 0.015:
            self.vc.harvest_facet(audio, style)

        # Repeat detection & ABA
        repeat_audio = self.wait_for_repeat()
        if repeat_audio is not None:
            try:
                segs, _ = self.model.transcribe(repeat_audio, language="en")
                repeat_text = " ".join(s.text for s in segs).strip().lower()
                if ratio(repeat_text, corrected_text) >= 0.93:
                    self.aba.track_skill_progress("articulation", success=True)
                    if self.aba.success_streak >= 3:
                        self.vc.say("Perfect! I’m so proud!", style="excited")
                else:
                    self.aba.track_skill_progress("articulation", success=False)
            except Exception as e:
                logging.error(f"Repeat processing failed: {e}")

    except Exception as e:
        logging.error(f"Utterance processing failed: {e}\n{traceback.format_exc()}")

def wait_for_repeat(self, timeout=6.0) -> np.ndarray | None:
    start = time.time()
    repeat_buffer = np.array([], dtype='float32')
    while time.time() - start < timeout:
        try:
            chunk = self.q.get(timeout=0.1)
            repeat_buffer = np.append(repeat_buffer, chunk)
        except queue.Empty:
            continue
    return repeat_buffer if len(repeat_buffer) > CONFIG.sample_rate * 0.5 else None

