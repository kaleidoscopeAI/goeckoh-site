def enforce_first_person(text: str) -> str:
    s = text.strip()
    if s and not s.endswith((".", "!", "?")):
        s += "."
    for pattern, repl in _FIRST_PERSON_RULES:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s


