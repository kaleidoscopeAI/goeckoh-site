Added from HID_AI_part_ab.txt and ac.txt: Emulates mouse/keyboard for companion interactions.
Pythonfrom __future__ import annotations

import time
import random
from typing import Optional

class HIDController:
    def __init__(self, device_path: str = '/dev/hidg0'):
        self.device_path = device_path
        # In real system, open HID gadget
        print(f"HID Controller initialized at {device_path}")

    def send_hid_report(self, report: bytearray):
        # Simulate sending; in real, write to device
        print(f"[HID] Sending report: {list(report)}")

    def move_mouse(self, dx: int, dy: int):
        report = bytearray([0x00, dx & 0xFF, dy & 0xFF, 0x00])
        self.send_hid_report(report)

    def mouse_click(self, button: int = 1):
        report = bytearray([button, 0x00, 0x00, 0x00])
        self.send_hid_report(report)
        time.sleep(0.05)
        report[0] = 0x00
        self.send_hid_report(report)

    def type_text(self, text: str):
        for char in text:
            # Simulate key press (simplified, real would use keycodes)
            key_code = ord(char) - 32  # Dummy
            report = bytearray([0x00, 0x00, key_code, 0x00, 0x00, 0x00, 0x00, 0x00])
            self.send_hid_report(report)
            time.sleep(0.02)
            report[2] = 0x00
            self.send_hid_report(report)
            time.sleep(random.uniform(0.05, 0.15))  # Mimic human typing

