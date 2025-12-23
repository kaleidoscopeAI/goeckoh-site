class GuidanceLogger:
    path: Path
    events: List[GuidanceEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.events.append(
                    GuidanceEvent(
                        timestamp=float(row["timestamp"]),
                        event_type=row["event_type"],
                        title=row["title"],
                        message=row["message"],
                    )
                )

    def log(self, event_type: EventType, title: str, message: str) -> None:
        ts = time.time()
        event = GuidanceEvent(timestamp=ts, event_type=event_type, title=title, message=message)
        self.events.append(event)
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "event_type", "title", "message"])
            if self.path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow({"timestamp": ts, "event_type": event_type, "title": title, "message": message})

