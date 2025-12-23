from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .models import AttemptRecord, BehaviorEvent


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _timestamp(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime(ISO_FORMAT)


class MetricsLogger:
    """Append-only CSV writer for attempt records."""

    header = (
        "timestamp",
        "phrase_text",
        "raw_text",
        "corrected_text",
        "needs_correction",
        "similarity",
        "audio_file",
    )

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def append(self, record: AttemptRecord) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    _timestamp(record.timestamp),
                    record.phrase_text,
                    record.raw_text,
                    record.corrected_text,
                    "1" if record.needs_correction else "0",
                    f"{record.similarity:.4f}",
                    str(record.audio_file),
                )
            )

    def tail(self, limit: int = 50) -> list[AttemptRecord]:
        """Return the last `limit` attempt records from the CSV, if present."""
        if not self.csv_path.exists():
            return []
        rows: list[AttemptRecord] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                except Exception:
                    continue
                rows.append(
                    AttemptRecord(
                        timestamp=ts,
                        phrase_text=row.get("phrase_text") or "",
                        raw_text=row.get("raw_text") or "",
                        corrected_text=row.get("corrected_text") or "",
                        needs_correction=(row.get("needs_correction") == "1"),
                        audio_file=Path(row.get("audio_file") or ""),
                        similarity=float(row.get("similarity") or 0.0),
                    )
                )
        return rows[-limit:]


class GuidanceLogger:
    """Structured log for behavior / guidance events."""

    header = ("timestamp", "level", "category", "title", "message", "metadata_json")

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def append(self, event: BehaviorEvent) -> None:
        payload = (
            _timestamp(event.timestamp),
            event.level,
            event.category,
            event.title,
            event.message,
            json.dumps(event.metadata, ensure_ascii=False),
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(payload)

    def tail(self, limit: int = 50) -> list[BehaviorEvent]:
        if not self.csv_path.exists():
            return []
        rows: list[BehaviorEvent] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                except Exception:
                    continue
                rows.append(
                    BehaviorEvent(
                        timestamp=ts,
                        level=row["level"],
                        category=row["category"],  # type: ignore[arg-type]
                        title=row["title"],
                        message=row["message"],
                        metadata=json.loads(row.get("metadata_json") or "{}"),
                    )
                )
        return rows[-limit:]

