"""Caregiver-facing reporting utilities."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import CompanionConfig


@dataclass(slots=True)
class SummaryRow:
    attempts: int = 0
    corrections: int = 0


def summarize(config: CompanionConfig) -> Dict[str, SummaryRow]:
    results: Dict[str, SummaryRow] = defaultdict(SummaryRow)
    path = config.paths.metrics_csv
    if not path.exists():
        return {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("phrase_id") or "<unknown>"
            stats = results[key]
            stats.attempts += 1
            if row.get("needs_correction") == "1":
                stats.corrections += 1
    return results
