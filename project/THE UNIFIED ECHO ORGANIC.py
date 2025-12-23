"""
THE UNIFIED ECHO-ORGANIC SYSTEM LAUNCHER
========================================
Combines:
1. The "Echo" Autism AGI (Real-time Speech/ABA)
2. The "Organic" Evolutionary Core (Background Node Processing)
3. The User Interfaces (Dashboard + GUI + CLI)

Run this file to start the ENTIRE system.
"""
import os
import sys
import time
import threading
import multiprocessing
import signal
from pathlib import Path

# --- Import Core Modules ---
# (Assuming your 72 scripts are in the python path. 
# If they are flat in one folder, these imports work directly.)
try:
    import config
    from speech_loop import SpeechLoop
    from agi_seed import AGICore  # The decision metabolism
    from dashboard import create_app as create_dashboard_app
    from tk_gui import EchoGUI  # The Tkinter Window
    from organic_core import OrganicSystem  # (Hypothetical wrapper for the Organic Node scripts)
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Missing System Modules: {e}")
    print("Ensure all 72 scripts are in this directory or properly installed packages.")
    sys.exit(1)

# --- Configuration ---
HOST_IP = "0.0.0.0"
DASHBOARD_PORT = 5000

class UnifiedSystem:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.organic_brain = None
        self.echo_loop = None
        self.gui = None
        
    def start_dashboard(self):
        """Starts the Flask Web Dashboard in a separate thread."""
        print("[System] Launching Caregiver Dashboard...")
        app = create_dashboard_app()
        # Disable reloader to prevent main thread interference
        app.run(host=HOST_IP, port=DASHBOARD_PORT, debug=False, use_reloader=False)

    def start_organic_background(self):
        """Starts the Self-Evolving 'Organic' Node System in background."""
        print("[System] Awakening Organic Core (Subconscious)...")
        # Initialize the Organic Node Manager
        # This assumes the organic scripts have a main controller.
        # If not, we initialize the seed nodes here.
        try:
            # Simulating the organic start based on your documents
            self.organic_brain = OrganicSystem(root_path="./organic_memory")
            self.organic_brain.start_metabolism() # Starts the cycle of node replication
        except Exception as e:
            print(f"[System] Organic Core failed to start (Non-Critical): {e}")

    def run(self):
        print("\n" + "="*60)
        print("       ECHO AGI : UNIFIED ORGANIC SYSTEM       ")
        print("="*60)
        
        # 1. Start the Subconscious (Organic AI)
        organic_thread = threading.Thread(target=self.start_organic_background, daemon=True)
        organic_thread.start()

        # 2. Start the Caregiver Dashboard (Flask)
        dashboard_thread = threading.Thread(target=self.start_dashboard, daemon=True)
        dashboard_thread.start()

        # 3. Initialize the Conscious Loop (Speech/ABA)
        print("[System] Initializing Conscious Speech Loop...")
        self.echo_loop = SpeechLoop()
        
        # 4. Launch the GUI (Must be on the Main Thread for Tkinter/Mac compatibility)
        print("[System] Opening GUI Interface...")
        try:
            # We pass the loop to the GUI so the GUI can visualize it
            self.gui = EchoGUI(loop_instance=self.echo_loop)
            
            # The GUI usually has a .mainloop() which blocks.
            # We start the speech loop in a background thread so it runs WHILE the GUI is open.
            speech_thread = threading.Thread(target=self.echo_loop.run_forever, args=(self.shutdown_event,), daemon=True)
            speech_thread.start()
            
            print(f"[System] SYSTEM LIVE. Dashboard: http://localhost:{DASHBOARD_PORT}")
            self.gui.root.mainloop() # Blocking call - runs until window closes
            
        except KeyboardInterrupt:
            print("\n[System] Shutdown signal received.")
        finally:
            self.shutdown_event.set()
            print("[System] Shutting down...")
            sys.exit(0)

if __name__ == "__main__":
    sys = UnifiedSystem()
    sys.run()
