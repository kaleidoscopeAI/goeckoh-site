def enforce_first_person(text: str) -> str:
    """Transform second-person phrasing into first person."""
    if not text: return ""
    t = text.strip()
    # Strip quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
        
    for pattern, repl in _FIRST_PERSON_PATTERNS:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return t

