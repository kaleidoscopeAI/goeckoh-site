class RoutineEngine:
    def __init__(self, paths: PathsConfig):
        self.paths = paths
        self.routines: List[Dict[str, Any]] = []
        self.last_spoken: Dict[str, float] = {}
        self._load_or_create_default()

    def _load_or_create_default(self) -> None:
        if self.paths.routine_file.exists():
            try:
                with open(self.paths.routine_file, "r", encoding="utf-8") as f:
                    self.routines = json.load(f)
                    print(f"[ROUTINE] Loaded {len(self.routines)} routines.")
                    return
            except Exception as e:
                print(f"[ROUTINE] Failed to load routine.json: {e}")

        self.routines = [
            {"id": "morning", "at_seconds": 8 * 3600, "text": "I wake up, stretch, and drink some water."},
            {"id": "meds", "at_seconds": 8 * 3600 + 1800, "text": "I remember to take my medicine calmly."},
            {"id": "evening", "at_seconds": 20 * 3600, "text": "I start winding down and I feel safe for sleep."},
        ]
        with open(self.paths.routine_file, "w", encoding="utf-8") as f:
            json.dump(self.routines, f, indent=2)
        print("[ROUTINE] Created default routine.json")

    def check_due(self, now: float) -> Optional[str]:
        local_time = time.localtime(now)
        seconds_today = local_time.tm_hour * 3600 + local_time.tm_min * 60 + local_time.tm_sec
        for routine in self.routines:
            rid = routine["id"]
            target = routine["at_seconds"]
            if abs(seconds_today - target) < 5 * 60:
                last = self.last_spoken.get(rid, 0.0)
                if now - last > 2 * 3600:
                    self.last_spoken[rid] = now
                    return routine["text"]
        return None


