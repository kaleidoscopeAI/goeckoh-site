def __init__(self):
    self.allow_hid = bool(int(os.environ.get("ALLOW_HID", "0")))
    if self.allow_hid:
        log.warning("ALLOW_HID enabled: Real HID actions. Ensure ECF verified.")
        try:
            self.device = evdev.UInput.from_device(evdev.list_devices()[0], name='Kaleidoscope-HID')
        except Exception as e:
            log.error(f"HID init failed: {e}. Falling back to simulated.")
            self.allow_hid = False

def execute_sequence(self, seq):
    for a in seq:
        op = a.get("op")
        delay = a.get("delay", 0.01)
        if self.allow_hid:
            if op == "key_press":
                key_code = getattr(evdev.ecodes, f'KEY_{a.get("key").upper()}', evdev.ecodes.KEY_A)
                self.device.write(evdev.ecodes.EV_KEY, key_code, 1)
                self.device.syn()
                time.sleep(0.01)
                self.device.write(evdev.ecodes.EV_KEY, key_code, 0)
                self.device.syn()
            log.info(f"EXEC HID {op} {a}")
        else:
            log.info(f"[SIMULATED HID] {op} {a}")
        time.sleep(delay)

