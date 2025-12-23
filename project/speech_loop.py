from __future__ import annotations

import argparse
import asyncio
import base64
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import CompanionConfig, CONFIG
from .audio_io import AudioIO
from .speech_processing import SpeechProcessor
from .data_store import DataStore, Phrase
from .similarity import SimilarityScorer
from .behavior_monitor import BehaviorMonitor
from .calming_strategies import StrategyAdvisor
from .advanced_voice_mimic import VoiceCrystal, VoiceCrystalConfig, VoiceProfile
from .voice_mimic import VoiceMimic
from .inner_voice import InnerVoiceEngine, InnerVoiceConfig
from .guidance import GuidanceCoach
from .text_utils import normalize_simple, text_similarity
from .agent import KQBCAgent
from .heart_core import EchoCrystallineHeart, enforce_first_person
from .events import EchoEvent, HeartMetrics, BrainMetrics, now_ts
from .core import CrystalBrain
from .trigger_engine import TriggerEngine
from .phenotyping_tracker import PhenotypingTracker
from .system_state import SystemState
from .controller import AvatarController
from .cca_bridge import CCABridgeClient
from .policy import GuardianPolicy


@dataclass(slots=True)
class SpeechLoop:
    """
    Real-time Echo loop:
    - Listens from the microphone.
    - Runs STT + grammar normalization.
    - Logs attempts + behaviour events.
    - Drives the Crystalline Heart + Crystal Brain + AGI substrate.
    - Speaks back a first-person inner echo in the child's own voice.
    """

    config: CompanionConfig = field(default_factory=lambda: CONFIG)
    state: Optional[SystemState] = None
    avatar: Optional[AvatarController] = None
    mirror_only: bool = False

    def __post_init__(self) -> None:
        # Low-level audio + models
        self.audio_io = AudioIO(self.config.audio)
        self.processor = SpeechProcessor(self.config.speech)
        self.data = DataStore(self.config)
        self.similarity = SimilarityScorer(self.config.audio)

        # Voice crystal + inner echo
        self.voice_tts = VoiceMimic(self.config.speech)
        profile_dir = self.config.paths.voices_dir / "voice_profile"
        self.voice_profile = VoiceProfile(audio=self.config.audio, base_dir=profile_dir)
        self.voice_crystal = VoiceCrystal(
            tts=self.voice_tts,
            audio=self.config.audio,
            profile=self.voice_profile,
            config=VoiceCrystalConfig(sample_rate=self.config.audio.sample_rate),
        )
        self.inner_voice = InnerVoiceEngine(
            voice=self.voice_crystal,
            audio_io=self.audio_io,
            data_store=self.data,
            config=InnerVoiceConfig(),
        )
        self.coach = GuidanceCoach(self.voice_crystal, self.audio_io, self.data)

        self._last_heart_metrics: Optional[HeartMetrics] = None
        self.cca_bridge: Optional[CCABridgeClient] = None
        self._cca_turn_index: int = 0

        # Mirror-only mode skips the heavier AGI/Heart stack and just echoes back.
        if self.mirror_only:
            self.phrases = {}
            return

        # Behaviour + ABA guidance
        self.behavior = BehaviorMonitor(
            max_phrase_history=self.config.behavior.max_phrase_history,
            anxious_threshold=self.config.behavior.anxious_threshold,
            perseveration_threshold=self.config.behavior.perseveration_threshold,
            high_energy_rms=self.config.behavior.high_energy_rms,
        )
        self.advisor = StrategyAdvisor()

        # Crystalline Heart v4.0 (ODE lattice + LLM sentience port)
        self.heart = EchoCrystallineHeart(self.config.heart)
        self.brain = CrystalBrain()
        self.agi = KQBCAgent(self.config)

        # PPP / phenotyping
        self.triggers = TriggerEngine()
        self.phenotyper = PhenotypingTracker()

        # Optional Cognitive Crystal AI backend bridge (Deep Reasoning Core)
        # controlled via environment variables for now.
        base_url = os.environ.get("CCA_BASE_URL")
        if base_url:
            session_id = os.environ.get("SESSION_ID", self.config.child_id)
            device_id = os.environ.get("ECHO_DEVICE_ID", "echo_local_01")
            self.cca_bridge = CCABridgeClient(base_url, session_id, device_id)

        # Guardian policy (Echo safety + quiet hours) loaded from local file.
        policy_path = Path(__file__).resolve().parent / "guardian_policy.json"
        self.guardian_policy: Optional[GuardianPolicy] = None
        if policy_path.exists():
            self.guardian_policy = GuardianPolicy.load(policy_path)

        # Cached phrases
        self.phrases = {p.phrase_id: p for p in self.data.list_phrases()}

    def record_phrase(self, text: str, seconds: float) -> Phrase:
        """Record a canonical phrase the child will practice."""
        pid = f"phrase_{int(time.time())}"
        audio = self.audio_io.record_phrase(seconds)
        filepath = self.config.paths.voices_dir / f"{pid}_{int(time.time())}.wav"
        self.audio_io.save_wav(audio, filepath)
        phrase = Phrase(
            phrase_id=pid,
            text=text,
            audio_file=filepath,
            duration=seconds,
            normalized_text=normalize_simple(text),
        )
        self.data.save_phrase(pid, text, filepath, seconds)
        self.phrases[pid] = phrase
        return phrase

    async def handle_chunk(self, chunk: np.ndarray) -> None:
        """Process a single fixed-length audio chunk."""
        if self.mirror_only:
            await self._handle_chunk_mirror(chunk)
            return

        if chunk.size == 0:
            return

        rms = self.audio_io.rms(chunk)
        if rms < self.config.audio.silence_rms_threshold:
            return

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp = Path(tmp_path)

        raw, corrected = await self.processor.process(
            chunk, tmp, self.config.audio.sample_rate
        )
        tmp.unlink(missing_ok=True)

        raw_fp = enforce_first_person(raw or "")
        corrected_fp = enforce_first_person(corrected or raw_fp or "")
        normalized_attempt = normalize_simple(corrected_fp or raw_fp or "")

        # Find best matching phrase by text
        best: Optional[Phrase] = None
        best_text_score = 0.0
        for phrase in self.phrases.values():
            score = text_similarity(normalized_attempt, phrase.normalized_text)
            if score > best_text_score:
                best = phrase
                best_text_score = score

        # Persist attempt audio
        attempt_path = self.config.paths.voices_dir / f"attempt_{int(time.time())}.wav"
        self.audio_io.save_wav(chunk, attempt_path)

        # Audio similarity vs canonical phrase (if any)
        audio_score = 0.0
        if best and best_text_score >= 0.4:
            try:
                audio_score = self.similarity.compare(best.audio_file, attempt_path)
            except Exception:
                audio_score = 0.0

        needs_correction = audio_score < 0.85

        # Crystalline Heart v4.0 + Brain + AGI update
        echo_event = EchoEvent(
            timestamp=now_ts(),
            text_raw=raw_fp or "",
            text_clean=normalized_attempt or "",
            duration_s=float(len(chunk) / self.config.audio.sample_rate),
            lang="en",
            meta={
                "rms": rms,
                "audio_score": audio_score,
                "needs_correction": needs_correction,
            },
        )

        heart_state = self.heart.step(chunk, normalized_attempt or "")
        heart_metrics = self._build_heart_metrics(heart_state)
        self._last_heart_metrics = heart_metrics

        self.brain.log_echo_event(echo_event)
        brain_metrics = self.brain.anneal_and_measure()

        clarity = self._compute_clarity(raw_fp or "", normalized_attempt or "")
        stability = self._compute_stability(heart_metrics)
        self.brain.log_voice_profile(clarity, stability)

        # Step AGI substrate with the child utterance
        if normalized_attempt:
            self.agi.update_state(user_input=normalized_attempt)

        # Generate caption + avatar, update SystemState for GUIs
        caption = self.brain.generate_caption()
        avatar_frame = None
        if self.avatar is not None:
            avatar_frame = self.avatar.update_from_state(
                heart_metrics,
                brain_metrics,
                caption,
            )
        if self.state is not None and avatar_frame is not None:
            self.state.update(
                echo=echo_event,
                heart=heart_metrics,
                brain=brain_metrics,
                caption=caption,
                avatar=avatar_frame,
            )

        # Inner echo (first-person, own timbre) when appropriate, gated by GCL
        gating_zone = self._gating_zone_from_metrics(heart_metrics)
        if self.config.behavior.correction_echo_enabled:
            self.inner_voice.speak_corrected(
                corrected_text=corrected_fp,
                raw_text=raw_fp,
                prosody_source_wav=chunk,
                prosody_source_sr=self.config.audio.sample_rate,
            )

        # Passive Voice Crystal adaptation when attempts are strong
        if best and not needs_correction and audio_score >= 0.85:
            style = (
                "calm"
                if rms < self.config.audio.silence_rms_threshold * 2
                else "neutral"
            )
            self.voice_profile.maybe_adapt_from_attempt(
                attempt_wav=chunk,
                style=style,
                quality_score=audio_score,
            )

        # Log attempt to CSV
        self.data.log_attempt(
            phrase_id=best.phrase_id if best else None,
            phrase_text=best.text if best else None,
            attempt_audio=attempt_path,
            stt_text=raw_fp,
            corrected_text=corrected_fp,
            similarity=audio_score,
            needs_correction=needs_correction,
        )

        # Behaviour monitoring + guidance
        event = self.behavior.register(
            normalized_text=normalized_attempt,
            needs_correction=needs_correction,
            rms=rms,
        )
        if event:
            suggestions = self.advisor.suggest(event)
            if suggestions:
                print(f"[EVENT] {event} detected")
                for s in suggestions[:3]:
                    print(f" - {s.title}: {s.description}")
            # Speak a calming or supportive script in outer voice
            self.coach.speak(event)

        # PPP phenotyping + snippet logging
        try:
            snippet_id = self.triggers.store_snippet(
                audio=chunk,
                sample_rate=self.config.audio.sample_rate,
                t=echo_event.timestamp,
                asr_text=raw or "",
            )
            self.phenotyper.log_fragment(
                snippet_id=snippet_id,
                vad_score=1.0,
                asr_confidence=1.0 if raw else 0.0,
                trigger="ASR",
            )
        except Exception as e:
            print(f"[PPP] Failed to log snippet: {e}")

        # Optional bridge to Cognitive Crystal AI backend (Deep Reasoning Core)
        self._maybe_send_cca_packet(
            audio=chunk,
            raw_text=raw or "",
            clean_text=normalized_attempt or "",
            heart=heart_metrics,
            gating_zone=gating_zone,
        )

        # Optional debugging output
        print(
            f"[Echo] text='{normalized_attempt}' "
            f"clarity={clarity:.2f} stability={stability:.2f} "
            f"stress={heart_metrics.stress:.2f} harmony={heart_metrics.harmony:.2f}"
        )

    async def _handle_chunk_mirror(self, chunk: np.ndarray) -> None:
        """Simplified path: listen, transcribe, and echo back in the child's voice."""
        if chunk.size == 0:
            return

        rms = self.audio_io.rms(chunk)
        if rms < self.config.audio.silence_rms_threshold:
            return

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp = Path(tmp_path)

        raw = ""
        corrected = ""
        try:
            raw, corrected = await self.processor.process(
                chunk, tmp, self.config.audio.sample_rate
            )
        finally:
            tmp.unlink(missing_ok=True)

        corrected_fp = enforce_first_person(corrected or raw or "")
        normalized_attempt = normalize_simple(corrected_fp or raw or "")

        attempt_path = self.config.paths.voices_dir / f"mirror_{int(time.time())}.wav"
        self.audio_io.save_wav(chunk, attempt_path)

        phrase_to_echo = corrected_fp or raw or normalized_attempt
        wav = self.voice_crystal.say_inner(
            phrase_to_echo,
            style="calm",
            prosody_source_wav=chunk,
            prosody_source_sr=self.config.audio.sample_rate,
        )
        if wav.size > 0:
            threading.Thread(target=self.audio_io.play, args=(wav,), daemon=True).start()

        # Persist a lightweight log for caregivers/review.
        try:
            self.data.log_attempt(
                phrase_id=None,
                phrase_text=None,
                attempt_audio=attempt_path,
                stt_text=raw or "",
                corrected_text=corrected_fp or "",
                similarity=1.0,
                needs_correction=False,
            )
        except Exception as e:
            print(f"[Mirror] Failed to log attempt: {e}")

        print(f"[Mirror] text='{normalized_attempt}' rms={rms:.4f}")

    async def run(self) -> None:
        """Main loop: stream audio from microphone and handle chunks."""
        stream = self.audio_io.microphone_stream()
        chunk_generator = self._speech_segments(stream)
        # Optionally start CCA command poller in the background
        if self.cca_bridge is not None:
            asyncio.create_task(self._cca_command_poller())
        async for chunk in _async_iter(chunk_generator):
            await self.handle_chunk(chunk)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _compute_clarity(text_raw: str, text_clean: str) -> float:
        """Approximate speech clarity as normalized Levenshtein similarity."""

        def levenshtein(a: str, b: str) -> int:
            if a == b:
                return 0
            if len(a) < len(b):
                a, b = b, a
            if len(b) == 0:
                return len(a)
            prev_row = list(range(len(b) + 1))
            for i, c1 in enumerate(a):
                curr_row = [i + 1]
                for j, c2 in enumerate(b):
                    cost = 0 if c1 == c2 else 1
                    curr_row.append(
                        min(
                            prev_row[j + 1] + 1,
                            curr_row[j] + 1,
                            prev_row[j] + cost,
                        )
                    )
                prev_row = curr_row
            return prev_row[-1]

        a = text_raw.strip().lower()
        b = text_clean.strip().lower()
        if not a and not b:
            return 1.0
        dist = float(levenshtein(a, b))
        denom = float(max(len(a), len(b), 1))
        return max(0.0, 1.0 - dist / denom)

    @staticmethod
    def _compute_stability(heart_metrics: HeartMetrics) -> float:
        """
        Map Heart metrics to a [0,1] stability score.

        High confidence + moderate energy + low stress → stable.
        """
        stress = np.clip(heart_metrics.stress, 0.0, 1.0)
        harmony = np.clip(heart_metrics.harmony, 0.0, 1.0)
        confidence = np.clip(heart_metrics.confidence, 0.0, 1.0)
        energy = np.clip(heart_metrics.energy, 0.0, 2.0)

        energy_delta = abs(energy - 1.0)
        inv_energy_delta = 1.0 - np.clip(energy_delta, 0.0, 1.0)
        inv_stress = 1.0 - stress

        stability = (
            0.25 * float(inv_stress)
            + 0.25 * float(harmony)
            + 0.25 * float(confidence)
            + 0.25 * float(inv_energy_delta)
        )
        return float(np.clip(stability, 0.0, 1.0))

    def _gating_zone_from_metrics(self, heart_metrics: Optional[HeartMetrics]) -> str:
        """
        Simple GCL-based gating as described in the docs.

        Returns one of: "red", "yellow", "green", "blue".
        """
        if heart_metrics is None:
            return "green"
        gcl = float(np.clip(heart_metrics.harmony, 0.0, 1.0))
        if gcl < 0.5:
            return "red"
        if gcl < 0.7:
            return "yellow"
        if gcl < 0.9:
            return "green"
        return "blue"

    def _speech_segments(self, stream: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        """
        Convert mic frames into full utterances using an autism-tuned VAD:
        - long silence patience (vad_min_silence_ms)
        - tolerant of quiet speech (silence_rms_threshold)
        - never cuts off mid-utterance.
        """
        sr = int(self.config.audio.sample_rate)
        min_speech = int(self.config.audio.vad_min_speech_ms * sr / 1000)
        min_silence = int(self.config.audio.vad_min_silence_ms * sr / 1000)
        pad = int(self.config.audio.vad_speech_pad_ms * sr / 1000)
        silence_thresh = float(self.config.audio.silence_rms_threshold)

        pre_speech: list[np.ndarray] = []
        speech_buf: list[np.ndarray] = []
        in_speech = False
        silence_frames = 0

        for block in stream:
            audio = block.reshape(-1)
            rms = self.audio_io.rms(audio)
            voiced = rms >= silence_thresh

            if voiced:
                silence_frames = 0
                if not in_speech:
                    # include a short lead-in when speech starts
                    if pre_speech:
                        speech_buf.extend(pre_speech[-1:])
                    in_speech = True
                speech_buf.append(audio)
            else:
                if in_speech:
                    silence_frames += len(audio)
                    speech_buf.append(audio)
                    if silence_frames >= min_silence:
                        utterance = np.concatenate(speech_buf)
                        if len(utterance) >= min_speech:
                            yield utterance
                        speech_buf = []
                        pre_speech = []
                        in_speech = False
                        silence_frames = 0
                else:
                    # keep a small pad of recent audio
                    pre_speech.append(audio)
                    total = sum(len(x) for x in pre_speech)
                    while total > pad and pre_speech:
                        dropped = pre_speech.pop(0)
                        total -= len(dropped)

    @staticmethod
    def _build_heart_metrics(heart_state: dict) -> HeartMetrics:
        """
        Derive HeartMetrics from EchoCrystallineHeart state.

        EchoCrystallineHeart exposes:
        - raw_emotions: [num_nodes, num_channels] tensor
        - temperature: scalar
        We reuse the same aggregation logic as the simpler Heart:
        stress ~ |channel 2|, energy ~ channel 4, harmony via variance,
        confidence via inverse global variance.
        """
        emotions = heart_state.get("raw_emotions")
        if emotions is None:
            # Fallback: neutral metrics
            t = now_ts()
            return HeartMetrics(
                timestamp=t,
                stress=0.0,
                harmony=1.0,
                energy=0.0,
                confidence=1.0,
                temperature=float(heart_state.get("temperature", 1.0)),
            )

        E = emotions.detach().cpu().numpy()

        # Channel indices: 0:arousal, 1:valence, 2:safety, 3:curiosity, 4:resonance
        # We treat inverse safety as "stress" and resonance as "energy".
        safety = float(np.mean(np.abs(E[:, 2])))
        stress = np.clip(1.0 - safety, 0.0, 1.0)

        energy_raw = float(np.mean(E[:, 4]))
        energy = float(np.clip(energy_raw, 0.0, 2.0))

        std_over_nodes = np.std(E, axis=0)
        mean_std = float(np.mean(std_over_nodes))
        harmony_raw = 1.0 / (1.0 + mean_std)
        harmony = float(np.clip(harmony_raw, 0.0, 1.0))

        confidence_raw = 1.0 / (1.0 + float(np.var(E)))
        confidence = float(np.clip(confidence_raw, 0.0, 1.0))

        t = now_ts()
        return HeartMetrics(
            timestamp=t,
            stress=stress,
            harmony=harmony,
            energy=energy,
            confidence=confidence,
            temperature=float(heart_state.get("temperature", 1.0)),
        )

    def _maybe_send_cca_packet(
        self,
        audio: np.ndarray,
        raw_text: str,
        clean_text: str,
        heart: HeartMetrics,
        gating_zone: str,
    ) -> None:
        """
        Optionally send a sensory packet to the Cognitive Crystal AI backend.

        This is gated by Heart coherence (no packets in deep RED) and
        mirrors the packet structure used by echo_prime.py.
        """
        if self.cca_bridge is None or gating_zone == "red":
            return

        self._cca_turn_index += 1

        arousal = float(np.clip(heart.energy * 5.0, 0.0, 10.0))
        valence = 0.0  # Placeholder; could be derived from caption or future sentiment.

        packet = {
            "session_id": self.cca_bridge.session_id,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "speaker": "child",
            "raw_text": raw_text,
            "clean_text": clean_text,
            "language": "en",
            "emotion": {
                "arousal": arousal,
                "valence": valence,
                "confidence": float(heart.confidence),
            },
            "audio": {
                "sample_rate": int(self.config.audio.sample_rate),
                "format": "wav16",
                "base64": self._encode_audio(audio, self.config.audio.sample_rate),
            },
            "aba": {
                "skill_tag": "default",
                "prompt_level": "none",
                "reinforcement_tag": "none",
                "behavior_notes": "none",
            },
            "echo_meta": {
                "utterance_id": f"echo_utt_{self._cca_turn_index:06d}",
                "turn_index": int(self._cca_turn_index),
                "device_id": self.cca_bridge.device_id,
            },
        }

        self.cca_bridge.send_sensory_packet(packet)

    @staticmethod
    def _encode_audio(audio_buffer: np.ndarray, sr: int) -> str:
        """Encode float32 -1..1 audio into base64-encoded 16-bit PCM."""
        if audio_buffer.ndim > 1:
            audio_buffer = audio_buffer[:, 0]
        pcm16 = (
            np.clip(audio_buffer, -1.0, 1.0) * 32767.0
        ).astype("<i2")
        return base64.b64encode(pcm16.tobytes()).decode("ascii")

    async def _cca_command_poller(self) -> None:
        """Background task that polls the CCA backend for commands."""
        if self.cca_bridge is None:
            return
        last_ts = 0.0
        while True:
            try:
                cmds = self.cca_bridge.pull_commands(last_ts)
                for cmd in cmds:
                    ts = float(cmd.get("ts", 0.0))
                    last_ts = max(last_ts, ts)
                    for action in cmd.get("actions", []):
                        self._apply_cca_action(action)
            except Exception as e:
                print(f"[CCA] Command poll error: {e}")
            await asyncio.sleep(0.25)

    def _apply_cca_action(self, action: dict) -> None:
        """Apply a single action from the CCA backend."""
        a_type = action.get("type")
        gating_zone = self._gating_zone_from_metrics(self._last_heart_metrics)

        echo_policy = {}
        night_cfg = {}
        allow_calming = True
        corrections_enabled = True
        in_quiet = False
        if self.guardian_policy is not None:
            echo_policy = self.guardian_policy.raw.get("echo", {}) or {}
            allow_calming = bool(echo_policy.get("allow_calming_scripts", True))
            corrections_enabled = bool(echo_policy.get("corrections_enabled", True))
            night_cfg = echo_policy.get("night_mode_quiet_hours", {}) or {}
            if night_cfg.get("enabled"):
                start = night_cfg.get("start_local", "")
                end = night_cfg.get("end_local", "")
                now_str = datetime.now().strftime("%H:%M")
                if start and end:
                    if start <= end:
                        in_quiet = start <= now_str < end
                    else:
                        # Quiet hours wrap around midnight
                        in_quiet = now_str >= start or now_str < end

        if a_type == "speech":
            text = (action.get("text") or "").strip()
            if not text:
                return

            mode = action.get("mode") or ""
            # Guardian may disable mirror-correction style prompts.
            if mode == "mirror_correct" and not corrections_enabled:
                return
            if not allow_calming:
                return

            prosody = action.get("prosody") or {}
            style = prosody.get("style", "neutral")

            # In red zone or quiet hours, downgrade to calmer inner-style delivery.
            if gating_zone == "red" or in_quiet:
                wav = self.voice_crystal.say_inner(text, style="calm")
            else:
                wav = self.voice_crystal.say_outer(text, style=style)
            if wav.size > 0:
                threading.Thread(
                    target=self.audio_io.play, args=(wav,), daemon=True
                ).start()
                # Log for caregiver review.
                self.data.log_guidance_event(
                    event="cca_speech",
                    title="CCA Speech Action",
                    message=text,
                )

        elif a_type == "aba":
            params = action.get("parameters") or {}
            text = (params.get("text") or "").strip()
            if not text:
                return
            if not allow_calming:
                return

            style = params.get("style", "calm")
            if gating_zone == "red" or in_quiet:
                wav = self.voice_crystal.say_inner(text, style="calm")
            else:
                wav = self.voice_crystal.say_outer(text, style=style)
            if wav.size > 0:
                threading.Thread(
                    target=self.audio_io.play, args=(wav,), daemon=True
                ).start()
            # Log as a guidance event for caregiver review.
            self.data.log_guidance_event(
                event="cca_aba",
                title="CCA ABA Strategy",
                message=text,
            )


@dataclass
class SimulatedSpeechLoop:
    """
    Async simulator that mirrors the realtime loop but without microphones.

    Generates toy phrases and routes them through KQBCAgent to exercise the
    AGI substrate and CSV logging, useful for demos and tests.
    """

    config: CompanionConfig = field(default_factory=lambda: CONFIG)
    agent: Optional[KQBCAgent] = None
    vocabulary: Sequence[str] = field(
        default_factory=lambda: (
            "hello",
            "water",
            "thank you",
            "help",
            "good morning",
            "yes",
            "no",
        )
    )

    def __post_init__(self) -> None:
        self.agent = self.agent or KQBCAgent(self.config)

    async def run(self) -> None:
        from .data_store import DataStore

        data = DataStore(self.config)
        vocab = list(self.vocabulary)
        while True:
            phrase = np.random.choice(vocab)
            raw = self._maybe_mutate_phrase(phrase)
            needs_correction = self.agent.evaluate_correction(phrase, raw)
            corrected = phrase if needs_correction else raw

            timestamp = datetime.utcnow().isoformat()
            attempts_path = self.config.paths.metrics_csv
            attempts_path.parent.mkdir(parents=True, exist_ok=True)
            import csv

            csv_exists = attempts_path.exists()
            with attempts_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    writer.writerow(
                        [
                            "timestamp_iso",
                            "child_id",
                            "phrase_id",
                            "phrase_text",
                            "attempt_audio",
                            "raw_text",
                            "corrected_text",
                            "similarity",
                            "needs_correction",
                        ]
                    )
                writer.writerow(
                    [
                        timestamp,
                        self.config.child_id,
                        "",
                        phrase,
                        "",
                        raw,
                        corrected,
                        f"{1.0 if not needs_correction else 0.0:.3f}",
                        "1" if needs_correction else "0",
                    ]
                )

            self.agent.update_state(user_input=raw)
            await asyncio.sleep(np.random.uniform(1.5, 3.0))

    @staticmethod
    def _maybe_mutate_phrase(phrase: str) -> str:
        if np.random.random() >= 0.3:
            return phrase
        if len(phrase) <= 2:
            return phrase
        idx = np.random.randint(0, len(phrase))
        if phrase[idx] in "aeiou":
            new_char = np.random.choice(list("aeiou"))
            return phrase[:idx] + new_char + phrase[idx + 1 :]
        return phrase[:idx] + phrase[idx + 1 :]


def _parse_args():
    parser = argparse.ArgumentParser(description="EchoVoice Speech Loop")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run without microphone; uses SimulatedSpeechLoop.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Run mirror-only mode (echo back without AGI stack).",
    )
    return parser.parse_args()


async def _run_simulated():
    sim = SimulatedSpeechLoop()
    await sim.run()


def main():
    args = _parse_args()
    if args.simulate:
        asyncio.run(_run_simulated())
        return
    loop = SpeechLoop(mirror_only=args.mirror)
    asyncio.run(loop.run())


async def _async_iter(sync_iter: Iterable[np.ndarray]):
    """Bridge a synchronous generator into an async for-loop."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            chunk = await loop.run_in_executor(None, next, sync_iter, None)
            if chunk is None:
                break
            yield chunk
        except StopIteration:
            break


if __name__ == "__main__":
    main()
