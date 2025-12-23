"""
ECHO AGI: TACTICAL CONTROL INTERFACE (GUI)
==========================================
A standalone, offline, dark-mode GUI for the Echo System.
Does NOT use a web browser.
Features:
- Real-time Arousal/Valence Biometrics
- Live 'Inner Voice' Transcription Log
- Manual ABA Override Controls
- Organic Core Visualization

Dependencies: pip install customtkinter
"""

import threading
import queue
import time
import tkinter as tk
import customtkinter as ctk
from datetime import datetime
from typing import Callable, Any, Optional

# Import your main system
try:
    from echo_prime import EchoSystem
except ImportError:
    print("CRITICAL: 'echo_prime.py' not found. GUI cannot link to Core.")
    # We will define a dummy class just so the GUI can launch and show you the layout
    class EchoSystem:
        def __init__(self):
            self.running = False
            self.heart = type('obj', (object,), {'arousal': 0.0, 'coherence': 1.0})() # Mock heart
            self.subconscious = type('obj', (object,), {'nodes': []})() # Mock subconscious
        def listen_loop(self):
            print("Dummy EchoSystem listen_loop called.")
            while self.running:
                # Simulate some activity for GUI testing without actual backend
                self.heart.arousal = (self.heart.arousal + 0.1) % 10.0
                self.heart.coherence = (self.heart.coherence + 0.01) % 1.0
                if random.random() < 0.1: # Simulate a new node
                    self.subconscious.nodes.append(type('obj', (object,), {'id': f"mock_node_{len(self.subconscious.nodes)}", 'energy': random.uniform(0.1, 10.0)})())
                if random.random() < 0.05 and self.subconscious.nodes: # Simulate node death
                    self.subconscious.nodes.pop(0)
                time.sleep(0.5)

        def set_gui_callback(self, callback_func: Callable[[str, Any], None]):
            self._gui_callback = callback_func

        def log(self, message_type: str, content: Any):
            if hasattr(self, '_gui_callback') and self._gui_callback:
                self._gui_callback(message_type, content)
            else:
                print(f"[{message_type}] {content}")


# --- UI Configuration ---
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light")
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue")

class EchoGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Window Setup
        self.title("ECHO AGI // MONITORING STATION")
        self.geometry("1100x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 2. Data Link
        self.system: Optional[EchoSystem] = None
        self.msg_queue = queue.Queue()
        self.running_system_thread = False

        # 3. Sidebar (System Control)
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="ECHO CORE\nSYSTEM V4", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.status_label = ctk.CTkLabel(self.sidebar, text="STATUS: OFFLINE", text_color="gray")
        self.status_label.grid(row=1, column=0, padx=20, pady=10)

        self.start_btn = ctk.CTkButton(self.sidebar, text="INITIALIZE SYSTEM", command=self.toggle_system, fg_color="green")
        self.start_btn.grid(row=2, column=0, padx=20, pady=10)

        # 4. Main Tabview
        self.tabs = ctk.CTkTabview(self, width=850)
        self.tabs.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        
        self.tab_monitor = self.tabs.add("LIVE FEED")
        self.tab_aba = self.tabs.add("ABA CONTROL")
        self.tab_organic = self.tabs.add("ORGANIC CORE")

        # --- TAB 1: LIVE MONITOR ---
        # Transcript Box
        self.transcript_label = ctk.CTkLabel(self.tab_monitor, text="INNER VOICE TRANSCRIPT", font=ctk.CTkFont(size=14, weight="bold"))
        self.transcript_label.pack(pady=(10,0), anchor="w", padx=10)
        
        self.console = ctk.CTkTextbox(self.tab_monitor, width=800, height=400, font=("Consolas", 14))
        self.console.pack(pady=10, padx=10, fill="both", expand=True)
        self.console.insert("0.0", "[SYSTEM READY] Awaiting initialization...\n")
        self.console.configure(state="disabled") # Read-only initially

        # Biometrics Bar (Arousal)
        self.bio_frame = ctk.CTkFrame(self.tab_monitor)
        self.bio_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(self.bio_frame, text="EMOTIONAL AROUSAL:").pack(side="left", padx=10)
        self.arousal_bar = ctk.CTkProgressBar(self.bio_frame, width=400, height=20)
        self.arousal_bar.set(0.0)
        self.arousal_bar.pack(side="left", padx=10)
        self.arousal_val = ctk.CTkLabel(self.bio_frame, text="0.0")
        self.arousal_val.pack(side="left", padx=5)

        # --- TAB 2: ABA CONTROL ---
        # Manual Override Buttons
        ctk.CTkLabel(self.tab_aba, text="MANUAL INTERVENTION OVERRIDES", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
        
        self.aba_btn_frame = ctk.CTkFrame(self.tab_aba)
        self.aba_btn_frame.pack(pady=10)

        self.btn_calm = ctk.CTkButton(self.aba_btn_frame, text="TRIGGER: CALMING", fg_color="#d97706", width=200, height=50, command=lambda: self.inject_command("calm"))
        self.btn_calm.grid(row=0, column=0, padx=20, pady=20)

        self.btn_focus = ctk.CTkButton(self.aba_btn_frame, text="TRIGGER: FOCUS", fg_color="#2563eb", width=200, height=50, command=lambda: self.inject_command("focus"))
        self.btn_focus.grid(row=0, column=1, padx=20, pady=20)
        
        self.btn_sleep = ctk.CTkButton(self.aba_btn_frame, text="MODE: SLEEP", fg_color="#4b5563", width=200, height=50, command=lambda: self.inject_command("sleep"))
        self.btn_sleep.grid(row=0, column=2, padx=20, pady=20)

        # Strategy Log
        ctk.CTkLabel(self.tab_aba, text="Active Strategy Log:").pack(anchor="w", padx=20)
        self.aba_log = ctk.CTkTextbox(self.tab_aba, height=200)
        self.aba_log.pack(fill="x", padx=20, pady=10)

        # --- TAB 3: ORGANIC CORE ---
        self.node_label = ctk.CTkLabel(self.tab_organic, text="SUBCONSCIOUS NODES ACTIVE: 0", font=ctk.CTkFont(size=24))
        self.node_label.pack(pady=40)
        
        self.coherence_label = ctk.CTkLabel(self.tab_organic, text="HEART COHERENCE: 100%", font=ctk.CTkFont(size=18))
        self.coherence_label.pack(pady=10)

        # DNA Mutation Rate Visualization
        ctk.CTkLabel(self.tab_organic, text="Genetic Drift (Mutation Rate)").pack(pady=(20,5))
        self.mutation_slider = ctk.CTkSlider(self.tab_organic, from_=0, to=1, number_of_steps=10)
        self.mutation_slider.set(0.1)
        self.mutation_slider.pack(pady=5)
        
        # 5. Periodic Update Loop
        self.after(100, self.update_gui)

    def gui_callback(self, message_type: str, content: Any):
        """Callback to receive messages from the EchoSystem thread."""
        self.msg_queue.put((message_type, content))

    def toggle_system(self):
        if not self.running_system_thread:
            # Start System
            self.running_system_thread = True
            self.start_btn.configure(text="SHUTDOWN SYSTEM", fg_color="red")
            self.status_label.configure(text="STATUS: ACTIVE", text_color="#22c55e")
            self.console.configure(state="normal")
            self.console.insert("end", "\n[GUI] INITIALIZING ECHO PRIME ENGINE...\n")
            self.console.configure(state="disabled")
            
            # Launch the Echo Thread
            self.thread = threading.Thread(target=self.run_echo_thread, daemon=True)
            self.thread.start()
        else:
            # Stop System (Simulated for safety)
            # In a real shutdown, you'd signal the EchoSystem thread to stop gracefully
            if self.system:
                self.system.running = False # Signal the dummy system to stop its loop
            self.running_system_thread = False
            self.start_btn.configure(text="INITIALIZE SYSTEM", fg_color="green")
            self.status_label.configure(text="STATUS: STOPPED", text_color="gray")
            self.console.configure(state="normal")
            self.console.insert("end", "\n[GUI] ECHO PRIME SYSTEM SHUTDOWN.\n")
            self.console.configure(state="disabled")

    def run_echo_thread(self):
        """Instantiates and runs the EchoSystem in a separate thread."""
        try:
            self.system = EchoSystem()
            self.system.set_gui_callback(self.gui_callback) # Set the callback for logging
            self.gui_callback("LOG", "Core Attached. Listening...")
            self.system.listen_loop() # This is a blocking call, needs to be in a thread
            self.gui_callback("LOG", "EchoSystem thread finished.")
        except Exception as e:
            self.gui_callback("ERROR", f"EchoSystem Thread Error: {e}")

    def inject_command(self, cmd: str):
        self.gui_callback("LOG", f"COMMAND INJECTED: {cmd.upper()}")
        # In real integration: self.system.inject_strategy(cmd) - you would add this method to EchoSystem

    def update_gui(self):
        """Polls the queue for updates from the AI thread and EchoSystem's internal state."""
        try:
            while True:
                type, data = self.msg_queue.get_nowait()
                
                if type == "LOG" or type == "TRANSCRIPT":
                    self.console.configure(state="normal")
                    self.console.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {data}\n")
                    self.console.see("end")
                    self.console.configure(state="disabled")
                
                elif type == "AROUSAL":
                    val = float(data)
                    self.arousal_bar.set(val / 10.0) # Assuming arousal is 0-10
                    self.arousal_val.configure(text=f"{val:.1f}")
                    if val > 7.0: self.arousal_bar.configure(progress_color="red")
                    elif val > 4.0: self.arousal_bar.configure(progress_color="orange")
                    else: self.arousal_bar.configure(progress_color="#3b8ed0")
                
                elif type == "ABA_STRATEGY":
                    self.aba_log.configure(state="normal")
                    self.aba_log.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {data}\n")
                    self.aba_log.see("end")
                    self.aba_log.configure(state="disabled")

                elif type == "ERROR":
                    self.console.configure(state="normal")
                    self.console.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {data}\n")
                    self.console.see("end")
                    self.console.configure(state="disabled")

        except queue.Empty:
            pass
        
        # Update Organic Core stats periodically, directly from the system instance
        if self.system and self.running_system_thread:
            # Check if the real system is running or if we're using the dummy
            if hasattr(self.system, 'heart') and hasattr(self.system.heart, 'arousal'):
                arousal_val = self.system.heart.arousal
                self.gui_callback("AROUSAL", arousal_val) # Re-send to update bar and color
            
            if hasattr(self.system, 'subconscious') and hasattr(self.system.subconscious, 'nodes'):
                node_count = len(self.system.subconscious.nodes)
                self.node_label.configure(text=f"SUBCONSCIOUS NODES ACTIVE: {node_count}")
            
            if hasattr(self.system, 'heart') and hasattr(self.system.heart, 'coherence'):
                coherence_val = self.system.heart.coherence
                self.coherence_label.configure(text=f"HEART COHERENCE: {coherence_val*100:.1f}%")

        self.after(100, self.update_gui) # Call itself again after 100ms

if __name__ == "__main__":
    import random # Only needed for dummy system simulation
    app = EchoGUI()
    app.mainloop()
