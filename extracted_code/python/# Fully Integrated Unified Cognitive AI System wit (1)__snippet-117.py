def _send_hid_report(self, report: bytearray):
    """Simulates sending a raw HID report to the host OS.
    In a real system, this would write bytes to a device file.
    """
    print(f"Simulating HID Report: {report.hex()} (to {self.device_path})")
    # Placeholder for actual device write
    pass 

def move_mouse(self, dx: int, dy: int):
    """Moves the mouse by a delta (dx, dy)."""
    report = bytearray(4) # [buttons, dx, dy, wheel]
    report[0] = 0x00  # No buttons pressed
    report[1] = dx & 0xFF # dx (signed byte)
    report[2] = dy & 0xFF # dy (signed byte)
    self._send_hid_report(report)
    # print(f"Mouse moved by ({dx}, {dy})")

def mouse_click(self, button: str = 'left'):
    """Performs a mouse click (press and release)."""
    button_code = 0x01 if button == 'left' else (0x02 if button == 'right' else 0x04) # Left, Right, Middle

    # Press button
    report = bytearray([button_code, 0, 0, 0])
    self._send_hid_report(report)
    time.sleep(0.05) # Small delay for click

    # Release button
    report = bytearray([0, 0, 0, 0])
    self._send_hid_report(report)
    # print(f"Mouse {button} clicked")

def key_press(self, key_code: int):
    """Presses and releases a single key."""
    report = bytearray(8) # [modifier, reserved, key1, key2, key3, key4, key5, key6]
    report[2] = key_code # Key code for the pressed key
    self._send_hid_report(report) # Press key
    time.sleep(0.05) # Small delay for key press

    report[2] = 0x00 # Release key
    self._send_hid_report(report)
    # print(f"Key code {key_code} pressed and released")

def type_string(self, text: str):
    """Simulates typing a string. Very basic, only supports alphanumeric and space.
    Requires a mapping from char to HID key code.
    """
    # This is a highly simplified mapping. Real HID would need a full keyboard layout.
    key_map = {
        'a': 0x04, 'b': 0x05, 'c': 0x06, 'd': 0x07, 'e': 0x08, 'f': 0x09, 'g': 0x0A, 'h': 0x0B, 
        'i': 0x0C, 'j': 0x0D, 'k': 0x0E, 'l': 0x0F, 'm': 0x10, 'n': 0x11, 'o': 0x12, 'p': 0x13, 
        'q': 0x14, 'r': 0x15, 's': 0x16, 't': 0x17, 'u': 0x18, 'v': 0x19, 'w': 0x1A, 'x': 0x1B, 
        'y': 0x1C, 'z': 0x1D, '1': 0x1E, '2': 0x1F, '3': 0x20, '4': 0x21, '5': 0x22, '6': 0x23, 
        '7': 0x24, '8': 0x25, '9': 0x26, '0': 0x27, ' ': 0x2C, '-': 0x2D, '=': 0x2E, '[': 0x2F,
        ']': 0x30, '\': 0x31, ';': 0x33, "'": 0x34, '`': 0x35, ',': 0x36, '.': 0x37, '/': 0x38,
        '\n': 0x28 # Enter key
    }
    for char in text.lower():
        if char in key_map:
            self.key_press(key_map[char])
            time.sleep(0.05) # Human-like typing speed
        else:
            print(f"Warning: Character '{char}' not supported by basic HID key map.")from .hid_controller import HIDController
class HardwareControl:
