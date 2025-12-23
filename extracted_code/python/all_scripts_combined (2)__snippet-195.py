class MeltdownRisk:
    score: int = 0  # 0–100
    level: str = "No data yet"
    message: str = ""


def _safe_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        return v in {"1", "true", "yes", "y"}
    return False


def load_metrics(config: CompanionConfig) -> MetricsSnapshot:
    path = config.paths.metrics_csv
    if path is None or not path.exists():
        return MetricsSnapshot()
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return MetricsSnapshot()
    total_attempts = len(rows)
    total_corrections = sum(1 for r in rows if _safe_bool(r.get("needs_correction")))
    last = rows[-1]
    last_phrase = last.get("phrase_text") or ""
    last_raw = last.get("raw_text") or ""
    last_corrected = last.get("corrected_text") or ""
    last_needs_correction = _safe_bool(last.get("needs_correction"))
    overall_rate = (total_corrections / total_attempts) if total_attempts else 0.0
    return MetricsSnapshot(
        total_attempts=total_attempts,
        total_corrections=total_corrections,
        overall_rate=overall_rate,
        last_phrase=last_phrase,
        last_raw=last_raw,
        last_corrected=last_corrected,
        last_needs_correction=last_needs_correction,
    )


def load_guidance(config: CompanionConfig) -> List[Dict[str, str]]:
    path = config.paths.guidance_csv
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_behavior_summary(
    guidance_rows: List[Dict[str, str]], window: int = 50
