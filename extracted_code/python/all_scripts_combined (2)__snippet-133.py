def enforce_first_person(text: str) -> str:
    """Transform any second-person phrasing into first person"""
    if not text:
        return text
    
    t = text.strip()
    
    # Strip quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    
    # Pattern replacements (case-insensitive)
    patterns = [
        (r"\byou are\b", "I am"),
        (r"\byou're\b", "I'm"),
        (r"\byou were\b", "I was"),
        (r"\byou'll\b", "I'll"),
        (r"\byou've\b", "I've"),
        (r"\byour\b", "my"),
        (r"\byours\b", "mine"),
        (r"\byourself\b", "myself"),
        (r"\byou can\b", "I can"),
        (r"\byou\b", "I"),
    ]
    
    for pattern, repl in patterns:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    
    return t

