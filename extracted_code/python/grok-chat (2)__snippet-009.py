def normalize_simple(text: str) -> str:
    text = re.sub(r\"[^a-z0-9 ]+\", \"\", text.lower()).strip()
    return text
