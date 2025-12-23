"""
Thin wrapper to ask LLM for action suggestions.
It must return a JSON with allowed keys.
We do no-code execution: parse-only.
"""
def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral", timeout: float = 6.0):
    self.base_url = base_url
    self.model = model
    self.timeout = timeout

def suggest_action(self, diag: Dict[str,float], extra: str = "") -> Optional[Dict[str,float]]:
    prompt = f"""Diagnostics: {json.dumps(diag)}.
