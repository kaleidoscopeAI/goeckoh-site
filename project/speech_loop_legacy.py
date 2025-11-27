# echo_core/speech_loop.py
# UPDATED: integrate voice_timbre + trigger_engine
from __future__ import annotations
import threading
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from TTS.api import TTS
from config import CONFIG
from events import EchoEvent, now_ts, HeartMetrics
from .speech_io import AudioStream, get_speech_prob
from .text_normalizer import normalize
from .heart import CrystallineHeart
from crystal_brain.core import CrystalBrain
from avatar.controller import AvatarController
from system_state import SystemState
from trigger_engine import TriggerEngine
from voice_timbre import update_auto_timbre_from_audio, get_auto_timbre_wav
from phenotyping_tracker import PhenotypingTracker

class SpeechLoop:
    """
    Unified core loop:
    - Listens with Silero VAD
    - Transcribes with faster-whisper
    - Normalizes to first-person
    - Updates Heart + Brain + Avatar
    - Uses child noises to grow a voice timbre
    - Uses guardian triggers to map noises -> first-person phrases
    - Speaks back ONLY first-person phrases in child-like voice
    - Publishes to SystemState for GUIs to consume
    """
    def __init__(
        self,
        state: SystemState,
        avatar: Optional[AvatarController] = None,
    ) -> None:
        self.audio_stream = AudioStream()
        self.heart = CrystallineHeart()
        self.brain = CrystalBrain()
        self.state = state
        self.avatar = avatar or AvatarController()
        self.triggers = TriggerEngine()
        self.phenotyper = PhenotypingTracker()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Lazy load models
        self._whisper: Optional[WhisperModel] = None
        self._tts: Optional[TTS] = None

    @property
    def whisper(self) -> WhisperModel:
        if self._whisper is None:
            print("Loading Whisper model...")
            self._whisper = WhisperModel(CONFIG.models.whisper_model, device="cpu", compute_type="int8")
        return self._whisper

    @property
    def tts(self) -> TTS:
        if self._tts is None:
            print("Loading TTS model...")
            self._tts = TTS(CONFIG.models.tts_model, progress_bar=False, gpu=False)
        return self._tts

    def _transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribes audio and returns the text and an average confidence score."""
        segments, info = self.whisper.transcribe(
            audio,
            language=CONFIG.profile.language,
            beam_size=1,
            vad_filter=False,
        )
        
        segment_list = list(segments) # Consume generator
        texts = [seg.text for seg in segment_list]
        full_text = " ".join(texts).strip()
        
        if not segment_list:
            return full_text, 0.0

        # Calculate confidence as the average of exp(avg_logprob) across segments
        confidences = [np.exp(s.avg_logprob) for s in segment_list if s.avg_logprob is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return full_text, float(avg_confidence)

    def _get_speaker_wav_path(self, heart_metrics: HeartMetrics) -> Optional[str]:
        mode = CONFIG.profile.voice_mode
        sample_dir = CONFIG.models.voice_samples_dir
        
        if mode == "true_child":
            target_type = "neutral"
            if heart_metrics.stress > 0.6 or heart_metrics.energy > 1.2:
                target_type = "excited"
            elif heart_metrics.harmony > 0.7 and heart_metrics.energy < 0.8:
                target_type = "tired"
            
            candidates = sorted(sample_dir.glob(f"jackson_{target_type}_*.wav"))
            if candidates:
                return str(candidates[0])
            any_jackson = sorted(sample_dir.glob("jackson_*.wav"))
            if any_jackson:
                return str(any_jackson[0])

        auto_path = get_auto_timbre_wav()
        if auto_path:
            return auto_path

        return CONFIG.profile.proxy_speaker_path

    def _play_tts(self, text: str, heart_metrics: HeartMetrics) -> float:
        mode = CONFIG.profile.voice_mode
        if mode == "silent":
            return 0.0

        speaker_wav = self._get_speaker_wav_path(heart_metrics)
        wav = None
        try:
            if speaker_wav:
                wav = self.tts.tts(text=text, speaker_wav=speaker_wav, language=CONFIG.profile.language)
            elif mode in ("proxy", "hybrid"):
                wav = self.tts.tts(text=text, speaker_wav=None, language=CONFIG.profile.language)
            else:
                return 0.0

            if wav is None: return 0.0

            wav_np = np.asarray(wav, dtype="float32")
            wav_np *= CONFIG.audio.inner_voice_volume
            
            delay_s = CONFIG.audio.inner_voice_delay_ms / 1000.0
            if delay_s > 0:
                time.sleep(delay_s)
                
            sd.play(wav_np, CONFIG.audio.sample_rate)
            sd.wait()
            return float(len(wav_np) / CONFIG.audio.sample_rate)
        except Exception as e:
            print(f"Error during TTS playback: {e}")
            return 0.0
    
    @staticmethod
    def _compute_clarity(text_raw: str, text_clean: str) -> float:
        def levenshtein(a: str, b: str) -> int:
            if a == b: return 0
            if len(a) < len(b): a, b = b, a
            if len(b) == 0: return len(a)
            prev_row = range(len(b) + 1)
            for i, c1 in enumerate(a):
                curr_row = [i + 1]
                for j, c2 in enumerate(b):
                    cost = 0 if c1 == c2 else 1
                    curr_row.append(min(prev_row[j+1]+1, curr_row[j]+1, prev_row[j]+cost))
                prev_row = curr_row
            return prev_row[-1]

        a = text_raw.strip().lower()
        b = text_clean.strip().lower()
        if not a and not b: return 1.0
        dist = float(levenshtein(a, b))
        denom = float(max(len(a), len(b), 1))
        return max(0.0, 1.0 - dist / denom)

    @staticmethod
    def _compute_stability(heart_metrics: HeartMetrics) -> float:
        stress = np.clip(heart_metrics.stress, 0.0, 1.0)
        harmony = np.clip(heart_metrics.harmony, 0.0, 1.0)
        confidence = np.clip(heart_metrics.confidence, 0.0, 1.0)
        energy = np.clip(heart_metrics.energy, 0.0, 2.0)
        energy_delta = abs(energy - 1.0)
        inv_energy_delta = 1.0 - np.clip(energy_delta, 0.0, 1.0)
        inv_stress = 1.0 - stress
        stability = 0.25 * inv_stress + 0.25 * harmony + 0.25 * confidence + 0.25 * inv_energy_delta
        return float(np.clip(stability, 0.0, 1.0))

    def _loop(self) -> None:
        self.audio_stream.start()
        buffer: list[np.ndarray] = []
        vad_scores: list[float] = []
        last_speech_time = None
        while not self._stop.is_set():
            chunk = self.audio_stream.get_chunk(timeout=0.5)
            if chunk is None: continue
            
            mono = chunk[:, 0] if chunk.ndim > 1 else chunk
            
            vad_score = get_speech_prob(mono)
            if vad_score > CONFIG.audio.vad_threshold:
                buffer.append(mono.copy())
                vad_scores.append(vad_score)
                last_speech_time = time.time()
            elif buffer and last_speech_time and (time.time() - last_speech_time) * 1000.0 > CONFIG.audio.vad_min_silence_ms:
                audio = np.concatenate(buffer, axis=0)
                avg_vad_score = np.mean(vad_scores) if vad_scores else 0.0
                
                buffer.clear()
                vad_scores.clear()
                last_speech_time = None
                
                threading.Thread(target=self._handle_utterance, args=(audio, avg_vad_score)).start()

    def _handle_utterance(self, audio: np.ndarray, avg_vad_score: float) -> None:
        start_ts = now_ts()
        update_auto_timbre_from_audio(audio, CONFIG.audio.sample_rate)
        
        text_raw, asr_confidence = self._transcribe(audio)
        
        snippet_id = self.triggers.store_snippet(
            audio=audio, sample_rate=CONFIG.audio.sample_rate, t=start_ts, asr_text=text_raw
        )

        trigger_source = "ASR"
        final_text = text_raw
        if not final_text.strip():
            match = self.triggers.match_for_audio(audio, CONFIG.audio.sample_rate)
            if match:
                final_text = match.phrase
                trigger_source = "Guardian"
        
        fragment_meta = self.phenotyper.log_fragment(
            snippet_id=snippet_id,
            vad_score=avg_vad_score,
            asr_confidence=asr_confidence,
            trigger=trigger_source
        )

        if not final_text.strip():
            return

        _, first_person = normalize(final_text)
        
        event = EchoEvent(
            timestamp=start_ts, text_raw=text_raw, text_clean=first_person,
            duration_s=float(len(audio) / CONFIG.audio.sample_rate),
            lang=CONFIG.profile.language,
            meta={
                "snippet_id": snippet_id,
                "phenotype": fragment_meta.classification.value,
                "asr_confidence": asr_confidence,
                "vad_score": avg_vad_score
            }
        )

        # === Core State Update ===
        heart_metrics = self.heart.update_from_event(event)
        self.brain.log_echo_event(event)
        brain_metrics = self.brain.anneal_and_measure()

        # === GCL-Based DRC Gating Logic ===
        gcl = heart_metrics.harmony
        drc_mode = "BLUE"
        caption = ""
        play_normal_tts = True
        calming_script = None

        if gcl < CONFIG.gating.RED_THRESHOLD:
            drc_mode = "RED"
            caption = "Feeling overwhelmed. Focusing on safety."
            play_normal_tts = False
            calming_script = "It's okay. I am safe. I can take a deep breath."
            print(f"DRC GATE: RED (GCL={gcl:.2f}). HALTING normal echo. Initiating calming script.")
        elif gcl < CONFIG.gating.YELLOW_THRESHOLD:
            drc_mode = "YELLOW"
            caption = self.brain.generate_caption()
            print(f"DRC GATE: YELLOW (GCL={gcl:.2f}). Internal focus. Limiting complex actions.")
        elif gcl < CONFIG.gating.GREEN_THRESHOLD:
            drc_mode = "GREEN"
            caption = self.brain.generate_caption()
            print(f"DRC GATE: GREEN (GCL={gcl:.2f}). Learning focus. Normal operation.")
        else: # gcl >= GREEN_THRESHOLD
            drc_mode = "BLUE"
            caption = self.brain.generate_caption()
            print(f"DRC GATE: BLUE (GCL={gcl:.2f}). Executive action. All systems nominal.")

        event.meta['drc_mode'] = drc_mode

        # === Final State Assembly & TTS ===
        clarity = self._compute_clarity(text_raw, first_person)
        stability = self._compute_stability(heart_metrics)
        self.brain.log_voice_profile(clarity, stability, start_ts)
        
        avatar_frame = self.avatar.update_from_state(heart=heart_metrics, brain=brain_metrics, caption=caption)
        
        self.state.update(echo=event, heart=heart_metrics, brain=brain_metrics, caption=caption, avatar=avatar_frame)
        
        # Execute TTS based on gating decision
        if calming_script:
            # Force a calm emotional state for the TTS when playing a calming script
            calm_metrics = HeartMetrics(
                timestamp=heart_metrics.timestamp, stress=0.1, harmony=0.9,
                energy=0.5, confidence=0.9, temperature=0.3
            )
            self._play_tts(calming_script, calm_metrics)
        elif play_normal_tts:
            self._play_tts(first_person, heart_metrics)

    def start(self) -> None:
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("SpeechLoop started.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.audio_stream.stop()
        print("SpeechLoop stopped.")
