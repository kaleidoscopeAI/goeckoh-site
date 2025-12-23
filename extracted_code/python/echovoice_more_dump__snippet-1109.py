"""Translates high-level AI 'Intent' into concrete HID actions."""
def __init__(self, hid_controller: HIDController):
    self.hid = hid_controller

def map_intent_to_action(self, intent: str):
    """Intent mapping based on Crystalline State."""
    if intent == "increase_performance":
        print("MAPPING: Intent 'increase_performance' -> HID: F11 (Fullscreen)")
        # Assuming key_code for F11 is 0x44 (HID usage code)
        self.hid.key_press(0x44) 
    elif intent == "defensive_stance_disconnect":
        print("MAPPING: Intent 'defensive_stance_disconnect' -> HID: Mouse movement")
        self.hid.move_mouse(50, 50) # Shift mouse cursor
    else:
        print(f"MAPPING: Intent '{intent}' -> No Action")

