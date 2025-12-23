def save_guidance(timestamp: float, line: str):
    line_fp = enforce_first_person(line)
    is_new = not GUIDANCE_CSV.exists()
    with GUIDANCE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "line"])
        writer.writerow([timestamp, line_fp])


def generate_guidance(raw_text: str, corrected_text: str) -> Optional[str]:
    if raw_text.strip().lower() == corrected_text.strip().lower():
        return None
    return f"I am getting better at saying: {corrected_text.strip()}"


