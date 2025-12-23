import re

def correct_text(raw_text: str, gcl: float):
    if not raw_text:
        return None
    
    # If stress is high (low GCL), repeat exactly what was said (safety mode)
    if gcl < 0.35:
        return raw_text.strip()
    
    txt = raw_text.strip()
    if not txt:
        return None
    
    # Simple expansion for single words
    if len(txt.split()) < 2:
        return f"I want the {txt}"
    
    # Pronoun flipping (The Mirror)
    rules = [
        (r"\b(A|a)re you\b", "am I"),
        (r"\b(Y|y)our\b", "my"),
        (r"\b(Y|y)ou\b", "I"),
    ]
    
    for p, r in rules:
        txt = re.sub(p, r, txt)
    
    # Grammar fixups
    txt = re.sub(r"\bI\s+are\b", "I am", txt)
    return txt
