def ask_llm_first_person(prompt: str, gcl: float) -> Optional[str]:
    if not SETTINGS.enable_llm or gcl < SETTINGS.gcl_high:
        return None
    payload = {"model": SETTINGS.llm_model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(SETTINGS.llm_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response") or data.get("output") or ""
    except Exception as e:
        print(f"[LLM] error: {e}")
        return None
    return enforce_first_person(text)


