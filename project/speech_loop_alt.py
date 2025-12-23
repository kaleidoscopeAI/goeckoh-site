"""Core speech companion loop with inner-voice echo + passive voice learning."""

from __future__ import annotations

import asyncio
import csv
import os
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from .advanced_voice_mimic import (
    VoiceCrystal,
    VoiceCrystalConfig,
    VoiceProfile,
)
from .audio_io import AudioIO, chunked_audio
from .behavior_monitor import BehaviorMonitor
from .calming_strategies import StrategyAdvisor
from .config import CompanionConfig
from .data_store import DataStore, Phrase
from .inner_voice import InnerVoiceEngine, InnerVoiceConfig
from .similarity import SimilarityScorer
from .speech_processing import SpeechProcessor
from .text_utils import normalize_simple, similarity as text_similarity
from .voice_mimic import VoiceMimic
from .guidance import GuidanceCoach
from .agent import KQBCAgent


@dataclass(slots=True)
class SpeechLoop:
    config: CompanionConfig

    def __post_init__(self) -> None:
        self.audio_io = AudioIO(self.config.audio)
        self.processor = SpeechProcessor(self.config.speech)
        self.data = DataStore(self.config)
        self.similarity = SimilarityScorer(self.config.audio)
        self.behavior = BehaviorMonitor()
        self.advisor = StrategyAdvisor()

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
            data_store=self.data,
            config=InnerVoiceConfig(),
        )
        self.coach = GuidanceCoach(self.voice_crystal, self.audio_io, self.data)

        self.phrases = {p.phrase_id: p for p in self.data.list_phrases()}

    def record_phrase(self, text: str, seconds: float) -> Phrase:
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
        if chunk.size == 0:
            return
        rms = self.audio_io.rms(chunk)
        if rms < self.config.audio.silence_rms_threshold:
            return

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp = Path(tmp_path)
        raw, corrected = await self.processor.process(chunk, tmp, self.config.audio.sample_rate)
        tmp.unlink(missing_ok=True)

        normalized_attempt = normalize_simple(corrected or raw or "")
        best: Optional[Phrase] = None
        best_text_score = 0.0
        for phrase in self.phrases.values():
            score = text_similarity(normalized_attempt, phrase.normalized_text)
            if score > best_text_score:
                best = phrase
                best_text_score = score

        attempt_path = self.config.paths.voices_dir / f"attempt_{int(time.time())}.wav"
        self.audio_io.save_wav(chunk, attempt_path)

        audio_score = 0.0
        if best and best_text_score >= 0.4:
            try:
                audio_score = self.similarity.compare(best.audio_file, attempt_path)
            except Exception:
                audio_score = 0.0

        needs_correction = audio_score < 0.85

        if needs_correction and self.config.behavior.correction_echo_enabled:
            self.inner_voice.speak_corrected(
                corrected_text=corrected,
                raw_text=raw,
                prosody_source_wav=chunk,
                prosody_source_sr=self.config.audio.sample_rate,
            )

        if best and not needs_correction and audio_score >= 0.85:
            style = "calm" if rms < self.config.audio.silence_rms_threshold * 2 else "neutral"
            self.voice_profile.maybe_adapt_from_attempt(
                attempt_wav=chunk,
                style=style,
                quality_score=audio_score,
            )

        self.data.log_attempt(
            phrase_id=best.phrase_id if best else None,
            phrase_text=best.text if best else None,
            attempt_audio=attempt_path,
            stt_text=raw,
            corrected_text=corrected,
            similarity=audio_score,
            needs_correction=needs_correction,
        )

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

    async def run(self) -> None:
        stream = self.audio_io.microphone_stream()
        chunk_generator = chunked_audio(
            stream,
            self.config.audio.chunk_seconds,
            self.config.audio.sample_rate,
        )
        async for chunk in _async_iter(chunk_generator):
            await self.handle_chunk(chunk)


async def _async_iter(sync_iter):
    loop = asyncio.get_event_loop()
    while True:
        try:
            chunk = await loop.run_in_executor(None, next, sync_iter, None)
            if chunk is None:
                break
            yield chunk
        except StopIteration:
            break


class SimulatedSpeechLoop:
    """Async simulator that mirrors the legacy speech_loop scripts.

    This loop does not access microphones. Instead it generates sample
    phrases, optionally routes them through ``KQBCAgent`` to update the
    cognitive substrate, and appends CSV logs identical to the real
    companion. Behavioural events are sampled from simple probabilities to
    keep downstream dashboards exercised during demos/tests.
    """

    def __init__(
        self,
        config: Optional[CompanionConfig] = None,
        *,
        agent: Optional[KQBCAgent] = None,
        vocabulary: Optional[Sequence[str]] = None,
        behaviour_events: Optional[Sequence[Tuple[str, float]]] = None,
        use_agent: bool = True,
    ) -> None:
        self.config = config or CompanionConfig()
        self.agent = agent if agent is not None else (KQBCAgent(self.config) if use_agent else None)
        self.vocabulary: List[str] = list(
            vocabulary
            or (
                "hello",
                "water",
                "thank you",
                "help",
                "good morning",
                "yes",
                "no",
            )
        )
        self.behaviour_events: List[Tuple[str, float]] = list(
            behaviour_events
            or (
                ("anxious", 0.05),
                ("perseveration", 0.03),
                ("high_energy", 0.04),
                ("meltdown", 0.01),
                ("encouragement", 0.06),
            )
        )

    async def run(self) -> None:
        """Run indefinitely, writing metrics + guidance rows."""

        self._ensure_metrics_header()
        self._ensure_guidance_header()
        while True:
            await self._simulate_one_turn()
            await asyncio.sleep(random.uniform(1.5, 3.0))

    async def _simulate_one_turn(self) -> None:
        phrase = random.choice(self.vocabulary)
        raw = self._maybe_mutate_phrase(phrase)

        if self.agent:
            needs_correction = self.agent.evaluate_correction(phrase, raw)
            corrected = phrase if needs_correction else raw
            self.agent.update_state(user_input=raw)
        else:
            needs_correction = raw != phrase
            corrected = phrase if needs_correction else raw

        timestamp = datetime.utcnow().isoformat()
        metrics_path = self.config.paths.metrics_csv
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                phrase,
                raw,
                corrected,
                "1" if needs_correction else "0",
            ])

        for event, prob in self.behaviour_events:
            if random.random() < prob:
                with self.config.paths.guidance_csv.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, event, phrase])
                break

    def _maybe_mutate_phrase(self, phrase: str) -> str:
        if random.random() >= 0.3:
            return phrase
        if len(phrase) <= 2:
            return phrase
        idx = random.randint(0, len(phrase) - 1)
        if phrase[idx] in "aeiou":
            new_char = random.choice("aeiou")
            return phrase[:idx] + new_char + phrase[idx + 1 :]
        return phrase[:idx] + phrase[idx + 1 :]

    def _ensure_metrics_header(self) -> None:
        path = self.config.paths.metrics_csv
        if not path.exists():
            path.touch()
        if path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "phrase_text",
                    "raw_text",
                    "corrected_text",
                    "needs_correction",
                ])

    def _ensure_guidance_header(self) -> None:
        path = self.config.paths.guidance_csv
        if not path.exists():
            path.touch()
        if path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "event",
                    "phrase_text",
                ])
