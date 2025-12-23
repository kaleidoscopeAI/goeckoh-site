import json
import time
from pathlib import Path
from typing import Any, Dict

from config import METRICS_DB_PATH


class MetricsStore:
    """
    Simple JSONL logger for attempts and timing.
    """

    def __init__(self, path: Path = METRICS_DB_PATH) -> None:
        self.path = path

    def log(self, record: Dict[str, Any]) -> None:
        entry = dict(record)
        entry.setdefault("ts", time.time())
        line = json.dumps(entry, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
