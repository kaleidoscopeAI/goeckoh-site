"""Load the current metrics from disk."""
path = config.paths.metrics_csv
if not path.exists():
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
return MetricsSnapshot(
    total_attempts=total_attempts,
    total_corrections=total_corrections,
    overall_rate=(total_corrections / total_attempts) if total_attempts else 0.0,
    last_phrase=last.get("phrase_text", ""),
    last_raw=last.get("raw_text", ""),
    last_corrected=last.get("corrected_text", ""),
    last_needs_correction=_safe_bool(last.get("needs_correction")),
)


