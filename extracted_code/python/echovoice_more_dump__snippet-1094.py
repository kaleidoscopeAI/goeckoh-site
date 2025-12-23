def __init__(self, config: Dict[str, Any] = None):
    cfg = config or {}
    self.l0_rules = cfg.get("L0", {})
    self.l1_rules = cfg.get("L1", {})
    self.l2_rules = cfg.get("L2", {})
    self.l3_rules = cfg.get("L3", {})
    self.l4_rules = cfg.get("L4", {})
    self.safe_mode = bool(int(os.environ.get("ICA_SAFE_MODE", "1")))

def check(self, intent: Dict[str, Any], uin_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons = []
    if intent.get("type") == "destroy_hardware":
        reasons.append("L0: cannot perform hardware-destruct intent")
    if intent.get("type") == "network" and intent.get("rate", 0) > 100 and self.l1_rules.get("rate_limit", True):
        reasons.append("L1: network rate too high")
    if intent.get("cost", 0) > 0.5 and uin_state.get("energy", 1.0) < 0.2:
        reasons.append("L2: insufficient energy")
    allowed = len(reasons) == 0
    if not allowed and self.safe_mode:
        raise GovernanceException("; ".join(reasons))
    return allowed, reasons

def penalty_for_violation(self, reasons: List[str]) -> Dict[str, float]:
    severity = 0.5 + 0.5 * min(len(reasons) / 5.0, 1.0)
    return {"delta_valence": -severity, "delta_arousal": min(severity, 1.0), "delta_energy": -0.05}

