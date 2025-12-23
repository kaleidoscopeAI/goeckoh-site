"""Emulates a Human Interface Device (HID) for low-level system control."""
def __init__(self, device_path='/dev/hidg0'):
    # In a real deployment, this path points to the HID gadget device
    self.device_path = device_path
    print(f"HID Controller initialized for path: {device_path}")

def send_hid_report(self, report_bytes):
    """Writes a low-level report to the HID device file."""
    print(f"HID Report Sent: {report_bytes}")
    # In a real environment, this line would be:
    # with open(self.device_path, 'wb') as f: f.write(report_bytes)

def move_mouse(self, dx, dy):
    """Translates high-level movement to a HID report."""
    # Standard mouse report format (buttons, x, y, scroll)
    report = bytearray([0x00, dx & 0xFF, dy & 0xFF, 0x00])
    self.send_hid_report(report)

def key_press(self, key_code):
    """Simulates a key press (e.g., for system commands)."""
    # Keyboard report format (modifier, reserved, keys[6])
    report = bytearray([0x00, 0x00] + [key_code] + [0x00] * 5)
    self.send_hid_report(report)
    time.sleep(0.01) # Release key
    report = bytearray([0x00] * 8)
    self.send_hid_report(report)

