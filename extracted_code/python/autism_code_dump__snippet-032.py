def compute_sentiment_score(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    return (pos - neg) / max(1, len(words))


def compute_meltdown_risk(gcl: float, rms: float, text: str) -> float:
    sent = compute_sentiment_score(text)
    gcl_term = 1.0 - min(1.0, max(0.0, gcl))
    rms_term = min(1.0, rms / 0.1)
    sent_term = 0.5 if sent < 0 else 0.0
    risk = 0.4 * gcl_term + 0.4 * rms_term + 0.2 * sent_term
    return min(1.0, max(0.0, risk))


