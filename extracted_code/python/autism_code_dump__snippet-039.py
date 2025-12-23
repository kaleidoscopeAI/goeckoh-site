def emit_haptic_pulse(duration_ms: int = 700):
    if not SETTINGS.enable_haptics or not HAPTICS_AVAILABLE:
        return
    try:
        vibrator.vibrate(duration=duration_ms)
    except Exception as e:
        print(f"[Haptics] error: {e}")


