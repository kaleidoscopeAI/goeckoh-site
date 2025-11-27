"""Persistence helpers for phrases, attempts, and caregiver metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import CompanionConfig
from .text_utils import normalize_simple


@dataclass(slots=True)
class Phrase:
    phrase_id: str
    text: str
    audio_file: Path
    duration: float
    normalized_text: str


@dataclass(slots=True)
class DataStore:
    """Thin abstraction around the folder layout described in the docs."""

    config: CompanionConfig
    metadata_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.config.paths.ensure()
        self.metadata_file = self.config.paths.voices_dir / f"{self.config.child_id}_phrases.json"

    def _load_metadata(self) -> Dict[str, dict]:
        if not self.metadata_file.exists():
            return {}
        return json.loads(self.metadata_file.read_text())

    def _save_metadata(self, meta: Dict[str, dict]) -> None:
        self.metadata_file.write_text(json.dumps(meta, indent=2))

    def list_phrases(self) -> List[Phrase]:
        phrases = []
        for pid, data in self._load_metadata().items():
            phrases.append(
                Phrase(
                    phrase_id=pid,
                    text=data["text"],
                    audio_file=Path(data["file"]),
                    duration=data.get("duration", 0.0),
                    normalized_text=data.get("normalized_text") or normalize_simple(data["text"]),
                )
            )
        return phrases

    def save_phrase(self, phrase_id: str, text: str, audio_file: Path, duration: float) -> None:
        meta = self._load_metadata()
        meta[phrase_id] = {
            "text": text,
            "file": str(audio_file),
            "duration": duration,
            "normalized_text": normalize_simple(text),
        }
        self._save_metadata(meta)

    def log_attempt(
        self,
        phrase_id: Optional[str],
        phrase_text: Optional[str],
        attempt_audio: Optional[Path],
        stt_text: str,
        corrected_text: str,
        similarity: float,
        needs_correction: bool,
    ) -> None:
        header = [
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
        new_row = [
            datetime.utcnow().isoformat(),
            self.config.child_id,
            phrase_id or "",
            phrase_text or "",
            str(attempt_audio) if attempt_audio else "",
            stt_text,
            corrected_text,
            f"{similarity:.3f}",
            "1" if needs_correction else "0",
        ]
        csv_exists = self.config.paths.metrics_csv.exists()
        with self.config.paths.metrics_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(header)
            writer.writerow(new_row)

    def log_guidance_event(self, event: str, title: str, message: str) -> None:
        header = ["timestamp_iso", "child_id", "event", "title", "message"]
        new_row = [
            datetime.utcnow().isoformat(),
            self.config.child_id,
            event,
            title,
            message,
        ]
        csv_exists = self.config.paths.guidance_csv.exists()
        with self.config.paths.guidance_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(header)
            writer.writerow(new_row)
