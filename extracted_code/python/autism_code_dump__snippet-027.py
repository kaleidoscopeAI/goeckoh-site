class SomaticEngine:
    def __init__(self):
        self.has_vibrator = HAS_VIBRATOR
        if self.has_vibrator:
            print("[SOMA] Haptic vibration enabled (plyer).")
        else:
            print("[SOMA] Haptic vibration not available on this platform.")

    def pulse(self, intensity: float, duration_ms: int = 500) -> None:
        if not self.has_vibrator:
            return
        try:
            duration = max(100, min(2000, duration_ms))
            vibrator.vibrate(time=duration / 1000.0)
        except Exception as e:
            print(f"[SOMA] Vibration error: {e}")


