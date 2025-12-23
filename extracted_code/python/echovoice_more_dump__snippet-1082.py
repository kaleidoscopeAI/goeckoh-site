def __init__(self):
    self.allow_hid = bool(int(os.environ.get("ALLOW_HID", "0")))
    if self.allow_hid:
        log.warning("ALLOW_HID enabled: Real HID actions. Ensure ECF verified.")
        try:
            self.device = evdev.UInput.from_device('/dev/input/event0', name='Kaleidoscope-HID')  # Replace with actual keyboard event
        except Exception as e:
            log.error(f"HID init failed: {e}. Falling back to sim.")
            self.allow_hid = False

def execute_sequence(self, seq):
    for a in seq:
        op = a.get("op")
        delay = a.get("delay", 0.01)
        if self.allow_hid:
            if op == "key_press":
                key = a.get("key")
                self.device.write(evdev.ecodes.EV_KEY, evdev.ecodes.KEY_A if key == 'a' else 0, 1)  # Real key press (example for 'a')
                self.device.syn()
                time.sleep(0.01)
                self.device.write(evdev.ecodes.EV_KEY, evdev.ecodes.KEY_A if key == 'a' else 0, 0)  # Release
                self.device.syn()
            log.info(f"EXEC HID {op} {a}")
        else:
            log.info(f"[SIMULATED HID] {op} {a}")
        time.sleep(delay)

