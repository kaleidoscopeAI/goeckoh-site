try:
    import pyautogui
    from pynput import keyboard
    HID_AVAILABLE=True
except Exception:
    HID_AVAILABLE=False
import time
from collections import deque
class HIDController:
    def __init__(self, safety_mode=True):
        self.safety_mode=safety_mode; self.action_history=deque(maxlen=256)
        self.min_delay=0.1; self.max_actions_per_minute=60; self.action_count=0; self.last=time.time(); self.stop=False
        if HID_AVAILABLE:
            self.w,self.h=pyautogui.size()
            def on_press(key):
                if key==keyboard.Key.f12: self.stop=True
            self.listener=keyboard.Listener(on_press=on_press,daemon=True); self.listener.start()
        else:
            self.w,self.h=(1920,1080)
    def _guard(self):
        if self.stop: raise RuntimeError("EMERGENCY_STOP")
        now=time.time()
        if now-self.last<self.min_delay: time.sleep(self.min_delay-(now-self.last))
        self.last=time.time(); self.action_count+=1
        if self.action_count>self.max_actions_per_minute: raise RuntimeError("RATE_LIMIT")
    def keyboard_type(self, text):
        if not HID_AVAILABLE: return False
        self._guard(); pyautogui.write(text, interval=0.05); return True
