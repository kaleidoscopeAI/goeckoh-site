def make_reflection(tick: int, m: Dict[str, float]) -> str:
    t = m.get("tension", 0.0)
    e = m.get("energy", 0.0)
    n = m.get("size", 0)
    mood = "calm" if t < 0.01 else "strained" if t < 0.04 else "overstretched"
    return f"[tick={tick}] Tension={t:.5f} Energy={e:.5f} Size={n}. State feels {mood}. Strategy: {'tighten springs' if t < 0.02 else 'loosen a bit' if t > 0.05 else 'hold steady'}."

def heuristic_adjust(m: Dict[str, float]) -> Dict[str, float]:
    t = m.get("tension", 0.0)
    if t < 0.015:
        return {"k_scale": 1.10, "rest_scale": 0.98}
    if t > 0.050:
        return {"k_scale": 0.95, "rest_scale": 1.03}
    return {"k_scale": 1.00, "rest_scale": 1.00}

def ask_ollama_refine(metrics: Dict[str, float], reflection: str) -> Dict[str, Any]:
    sys_p = "You optimize a spring-mesh cube. Reply STRICT JSON only: {\"k_scale\":float, \"rest_scale\":float}."
    user_p = f"Metrics: {metrics}\nReflection: {reflection}\nReturn only JSON."
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_MODEL, "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}], "stream": False}, timeout=15)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "").strip()
        data = json.loads(content)
        if not all(k in data for k in ("k_scale", "rest_scale")):
            raise ValueError("Missing keys")
        return {"ok": True, "adjust": data, "raw": content}
    except Exception as e:
        return {"ok": False, "adjust": heuristic_adjust(metrics), "error": str(e)}

