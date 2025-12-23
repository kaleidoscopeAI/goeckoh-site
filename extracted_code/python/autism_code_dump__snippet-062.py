def __init__(self, cfg: LLMConfig):
    self.cfg = cfg

def is_available(self) -> bool:
    if not self.cfg.enabled:
        return False
    from shutil import which

    return which("ollama") is not None

def answer_if_safe(self, prompt: str, gcl: float) -> Optional[str]:
    if not self.is_available():
        return None
    if gcl < self.cfg.gcl_threshold:
        return None
    try:
        proc = subprocess.run(
            ["ollama", "run", self.cfg.ollama_model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.cfg.timeout_sec,
        )
        if proc.returncode != 0:
            print(f"[LLM] Ollama error: {proc.stderr.decode('utf-8', errors='ignore')}")
            return None
        text = proc.stdout.decode("utf-8", errors="ignore").strip()
        text = text.replace(" you ", " I ").replace(" your ", " my ")
        return text
    except Exception as e:
        print(f"[LLM] Exception: {e}")
        return None


