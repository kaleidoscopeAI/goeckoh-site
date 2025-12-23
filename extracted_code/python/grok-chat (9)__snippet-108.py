252
253 +    def validation_summary(self) -> dict:
254 +        """Return concise stats for validation/telemetry endpoints."""
255 +        with self._lock:
256 +            latencies = [float(t.get("latency_ms", 0.0)) for t in self.
     _telemetry_hist]
257 +            gcls = [float(t.get("gcl", 0.0)) for t in self._telemetry_h
     ist]
258 +            drifts = [float(t.get("drift", 0.0)) for t in self._telemet
     ry_hist]
259 +            return {
260 +                "count": len(self._telemetry_hist),
261 +                "latency_p50_ms": self._percentile(latencies, 50.0),
262 +                "latency_p95_ms": self._percentile(latencies, 95.0),
263 +                "gcl_mean": float(sum(gcls) / len(gcls)) if gcls else 0
     .0,
264 +                "gcl_min": float(min(gcls)) if gcls else 0.0,
265 +                "drift_p95": self._percentile(drifts, 95.0),
266 +                "gate_blocks": int(self._gate_blocked),
267 +            }
268 +
269 +    def reset_metrics(self) -> None:
270 +        """Clear telemetry and fragment metrics (useful for validation
     runs)."""
271 +        with self._lock:
272 +            self._telemetry_hist.clear()
273 +            self._metrics_hist.clear()
274 +            self._drift_hist.clear()
275 +            self._metric_total_utts = 0
276 +            self._metric_latency_sum_ms = 0.0
277 +            self._metric_latency_max_ms = 0.0
278 +            self._gate_blocked = 0
279 +
280      def start(

