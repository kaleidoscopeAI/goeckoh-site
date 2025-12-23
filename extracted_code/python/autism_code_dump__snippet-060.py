def __init__(self, cfg: BehaviorConfig):
    self.cfg = cfg
    self.recent_gcl: List[float] = []
    self.recent_rms: List[float] = []
    self.recent_success: List[float] = []

def update(self, gcl: float, rms: float, success: float, text: str) -> Dict[str, Any]:
    self.recent_gcl.append(gcl)
    self.recent_rms.append(rms)
    self.recent_success.append(success)

    if len(self.recent_gcl) > self.cfg.meltdown_window:
        self.recent_gcl.pop(0)
        self.recent_rms.pop(0)
    if len(self.recent_success) > self.cfg.success_window:
        self.recent_success.pop(0)

    meltdown_risk = self._compute_meltdown_risk(text)
    success_level = float(sum(self.recent_success) / max(1, len(self.recent_success)))

    mode = "normal"
    if meltdown_risk > 0.7:
        mode = "meltdown_risk"
    elif success_level > 0.9:
        mode = "celebrate"

    return {
        "meltdown_risk": meltdown_risk,
        "success_level": success_level,
        "mode": mode,
    }

def _compute_meltdown_risk(self, text: str) -> float:
    if not self.recent_gcl:
        return 0.0
    avg_gcl = sum(self.recent_gcl) / len(self.recent_gcl)
    avg_rms = sum(self.recent_rms) / len(self.recent_rms)
    neg_count = sum(1 for w in self.cfg.negative_words if w in text)
    pos_count = sum(1 for w in self.cfg.positive_words if w in text)

    risk = 0.0
    if avg_gcl < self.cfg.meltdown_gcl_low:
        risk += 0.4
    if avg_rms > self.cfg.meltdown_rms_high:
        risk += 0.3
    if neg_count > 0:
        risk += 0.2
    if pos_count > 0:
        risk -= 0.2
    return max(0.0, min(1.0, risk))


