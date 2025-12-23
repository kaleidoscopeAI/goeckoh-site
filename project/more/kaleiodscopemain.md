# uni_core.py
# Main Python orchestration layer for the Unified Nexus Intelligence (UNI) System.
# Integrates the C-backend, LLM (Ollama), and the HID Emulation Layer for control.

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ctypes
import time
import os

# --- 1. Load C-Backend Library ---
try:
    # NOTE: You must compile kaleidoscope_core.c into a shared library first (e.g., libkaleidoscope.so)
    # Compilation example: gcc -shared -o libkaleidoscope.so -fPIC kaleidoscope_core.c
    LIB_PATH = os.path.join(os.path.dirname(__file__), "libkaleidoscope.so")
    if not os.path.exists(LIB_PATH):
        print(f"WARNING: C library not found at {LIB_PATH}. Using mock function.")
        # Fallback to mock purity function if C library is not available
        def compute_purity_mock(rho, N): return 0.5 + np.random.rand() * 0.5
        C_CORE = None
    else:
        C_CORE = ctypes.CDLL(LIB_PATH)
        # Define C function signatures for ctypes (essential for reliable calls)
        C_CORE.compute_purity.argtypes = [ctypes.c_void_p, ctypes.c_int]
        C_CORE.compute_purity.restype = ctypes.c_double
        print("C-Core (libkaleidoscope.so) loaded successfully.")

except Exception as e:
    print(f"ERROR LOADING C LIBRARY: {e}")
    C_CORE = None

# --- 2. UNI System State and Core Logic ---

app = FastAPI(
    title="Unified Nexus Intelligence Core",
    description="Geometric, Quantum-Integrated AI Core with Adaptive Cyber-Physical Control.",
)

# Mock Node/System State for demonstration
SYSTEM_STATE = {
    "node_count": 8,
    "nodes": [{"pos": [np.random.rand() * 10] * 3, "state": np.random.rand()} for _ in range(8)],
    "llm_summary": "System initialized and awaiting input.",
    "emotional_valence": 0.0,
    "system_purity": 1.0, # Target purity value
}

class Snapshot(BaseModel):
    node_data: list
    system_metrics: dict
    adc_raw: list = [] # Mock Analog-to-Digital Converter raw sensor data

# --- 3. HID Emulation Layer (L4 Cyber-Physical Control) ---

class HIDController:
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

class ControlMapper:
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

HID_CTRL = HIDController()
CONTROL_MAPPER = ControlMapper(HID_CTRL)

# --- 4. FastAPI Endpoints and Core System Steps ---

def sensor_grad(kappa, adc_raw):
    """Simulates the sensor gradient logic (∂I/∂t = ∇ · (κ ADC))"""
    if not adc_raw:
        return [0.0]
    # Simple discrete gradient calculation
    return np.gradient(np.array(adc_raw)) * kappa

@app.post("/speculate")
async def speculate_step(snapshot: Snapshot):
    """
    Core cognitive step: Ingests sensor data, calculates geometric flow,
    and updates system state based on the Forman-Ricci Flow (FRF) analog.
    """
    
    # 1. Process Sensor Data (Speculation Engine)
    kappa = snapshot.system_metrics.get("kappa", 1.0)
    sensor_update = sensor_grad(kappa, snapshot.adc_raw).tolist()

    # 2. Geometric Core (C-Backend Call)
    # Mock data for C-call (Density Matrix for Purity calculation)
    N = snapshot.system_metrics.get("node_count", SYSTEM_STATE["node_count"])
    # Create a mock flattened density matrix (N*N complex values)
    mock_rho_data = [complex(np.random.rand(), np.random.rand()) for _ in range(N*N)] 

    if C_CORE:
        # Prepare C types for complex array
        class ComplexArray_c(ctypes.Structure):
            _fields_ = [("n", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_double_complex))]

        rho_ptr = (ctypes.c_double_complex * len(mock_rho_data))(*mock_rho_data)
        
        # NOTE: C-side allocation would be preferred, but for simple mock passing:
        class ComplexArray_c_ptr(ctypes.Structure):
            _fields_ = [("n", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_double_complex))]
        
        rho_struct = ComplexArray_c_ptr(N * N, rho_ptr)
        
        # The structure of the C function call requires proper pointer handling
        # For simplicity, we assume compute_purity expects the array data pointer and size
        # A more robust solution involves creating a proper wrapper class
        
        # Fallback to direct call with pointer if structure is too complex for quick testing
        try:
             # A simpler call assuming C function takes a pointer to the data and the size
             purity = C_CORE.compute_purity(ctypes.cast(rho_ptr, ctypes.c_void_p), N)
        except Exception as e:
             print(f"C-Core function call failed: {e}. Falling back to mock purity.")
             purity = 0.5 + np.random.rand() * 0.5 # Mock Purity calculation
    else:
        purity = 0.5 + np.random.rand() * 0.5 # Mock Purity calculation

    # 3. Decision/Mapping (Control Engine)
    current_intent = "increase_performance" if purity < 0.6 else "defensive_stance_disconnect"
    CONTROL_MAPPER.map_intent_to_action(current_intent)

    # 4. Update Global State
    SYSTEM_STATE["system_purity"] = purity
    SYSTEM_STATE["emotional_valence"] += (purity - 0.7) * 0.1
    SYSTEM_STATE["llm_summary"] = f"Purity: {purity:.4f}. Intent: {current_intent}. Sensor Grad: {sensor_update[0]:.2f}."


    return {
        "status": "step_complete",
        "system_state": SYSTEM_STATE,
        "geometric_update": sensor_update,
        "purity_tr_rho_sq": purity,
    }

@app.get("/status")
def get_status():
    """Returns the current stable state of the system."""
    return SYSTEM_STATE

@app.get("/")
def read_root():
    return {"message": "UNI Core is operational. Use /speculate to advance the simulation."}
