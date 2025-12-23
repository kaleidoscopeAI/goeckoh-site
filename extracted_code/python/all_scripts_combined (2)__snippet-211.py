"""Append-only CSV writer for attempt records."""

header = ("timestamp", "phrase_text", "raw_text", "corrected_text", "needs_correction", "similarity", "audio_file")

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


