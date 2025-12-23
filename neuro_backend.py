import sherpa_onnx
import queue
import threading
import time
import numpy as np
from src.platform_utils import get_asset_path
from src.heart import CrystallineHeart
from src.grammar import correct_text
from src.config import load_config
from src.audio_manager import get_audio_driver
from src.logger import ClinicalLogger, logger

class NeuroKernel:
    def __init__(self, ui_queue=None):
        self.cfg = load_config()
        self.heart = CrystallineHeart()
        self.ui_queue = ui_queue
        self.mic_q = queue.Queue()
        self.spk_q = queue.Queue()
        self.csv_q = queue.Queue()
        
        self.logger_svc = ClinicalLogger(self.csv_q)
        self.logger_svc.start()

        try:
            self.driver = get_audio_driver(self.mic_q, self.spk_q, self.cfg["sample_rate"])
        except Exception as e:
            logger.error(f"Hardware Fail: {e}")
            self.driver = None

        self._load_ai()

    def _load_ai(self):
        logger.info("Loading Neural Network...")
        tokens = get_asset_path(self.cfg["tokens"])
        encoder = get_asset_path(self.cfg["encoder"])
        decoder = get_asset_path(self.cfg["decoder"])
        tts_model = get_asset_path(self.cfg["tts_model"])
        tts_tokens = get_asset_path(self.cfg["tts_tokens"])

        self.asr = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=encoder, decoder=decoder,
            num_threads=2, sample_rate=16000, feature_dim=80, enable_endpoint=True
        )
        self.asr_stream = self.asr.create_stream()
        
        self.tts = sherpa_onnx.OfflineTts(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=tts_model, tokens=tts_tokens)
            )
        )

    def start(self):
        if self.driver:
            self.driver.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
.
        while True:
            try:
                try:
                    chunk = self.mic_q.get(timeout=2.0)
                except queue.Empty:
                    continue

                x = chunk.flatten().astype(np.float32)
                rms = float(np.sqrt(np.mean(x ** 2)))
                
                # 1. Physics Pulse
                gcl, ent = self.heart.pulse(rms, 0.0)
                
                if self.ui_queue:
                    self.ui_queue.put(("PHY", {"gcl": gcl, "rms": rms, "ent": ent}))

                # 2. ASR
                self.asr_stream.accept_waveform(16000, x)
                if self.asr.is_endpoint(self.asr_stream):
                    txt = self.asr.get_result(self.asr_stream).strip()
                    self.asr.reset(self.asr_stream)
                    
                    if txt:
                        t0 = time.time()
                        
                        # 3. Mirror Logic
                        out = correct_text(txt, gcl)
                        if self.ui_queue:
                            self.ui_queue.put(("TXT", f"{txt} -> {out}"))
                        
                        # 4. Cloning/TTS
                        wav = self.tts.generate(out, sid=0, speed=1.0)
                        audio_out = np.array(wav.samples, dtype=np.float32).reshape(-1, 1)
                        
                        # 5. Soft Limiter (Safety)
                        peak = np.max(np.abs(audio_out))
                        if peak > 0.9:
                            audio_out *= 0.9 / peak
                        
                        self.spk_q.put(audio_out)
                        
                        # 6. Latency Calculation
                        lat = (time.time() - t0) * 1000.0
                        self.csv_q.put((rms, gcl, ent, lat, txt, out))

            except Exception as e:
                logger.error(f"Loop Crash: {e}")
