config: CompanionConfig

def list_phrases(self) -> List[Phrase]:
    rows = self._load_attempts(self.config.paths.attempts_csv)  # Reuse logic, but filter for canonical
    phrases = []
    seen = set()
    for row in rows:
        if row.get("is_canonical", "false") == "true" and row["phrase_id"] not in seen:
            phrases.append(Phrase(
                phrase_id=row["phrase_id"],
                text=row["phrase_text"],
                audio_file=Path(row["audio_file"]),
                duration=float(row["duration"]),
                normalized_text=row.get("normalized_text", "")
            ))
            seen.add(row["phrase_id"])
    return phrases

def save_phrase(self, phrase_id: str, text: str, audio_file: Path, duration: float, normalized_text: str) -> None:
    row = {
        "timestamp": time.time(),
        "phrase_id": phrase_id,
        "phrase_text": text,
        "audio_file": str(audio_file),
        "duration": duration,
        "normalized_text": normalized_text,
        "is_canonical": "true"
    }
    self._append_row(self.config.paths.attempts_csv, row)

def log_attempt(self, phrase_id: str | None, phrase_text: str | None, attempt_audio: Path, stt_text: str, corrected_text: str, similarity: float, needs_correction: bool) -> None:
    row = {
        "timestamp": time.time(),
        "phrase_id": phrase_id or "",
        "phrase_text": phrase_text or "",
        "attempt_audio": str(attempt_audio),
        "stt_text": stt_text,
        "corrected_text": corrected_text,
        "similarity": similarity,
        "needs_correction": needs_correction
    }
    self._append_row(self.config.paths.attempts_csv, row)

def log_guidance_event(self, event: str, title: str, message: str) -> None:
    row = {
        "timestamp": time.time(),
        "event": event,
        "title": title,
        "message": message
    }
    self._append_row(self.config.paths.guidance_csv, row)

def _load_attempts(self, path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(\"r\", newline=\"\") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _append_row(self, path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open(\"a\", newline=\"\") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
