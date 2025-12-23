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
            rows.append(
                BehaviorEvent(
                    timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
                    level=row["level"],
                    category=row["category"],  # type: ignore[arg-type]
                    title=row["title"],
                    message=row["message"],
                    metadata=json.loads(row.get("metadata_json") or "{}"),
                )
            )
    return rows[-limit:]

