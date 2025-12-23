def gating_zone_from_metrics(gcl: float) -> str:
    if gcl < SETTINGS.gcl_low:
        return "low"
    if gcl > SETTINGS.gcl_high:
        return "high"
    return "mid"


