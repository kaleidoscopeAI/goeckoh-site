def send_report(self, action):
    print(f"[HID Sim]: {action}")  # Sim mouse/type

def type_therapy(self, text):
    self.send_report(f"Typing: {text}")

