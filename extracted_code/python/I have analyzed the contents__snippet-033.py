def __init__(self):
    super().__init__()

    # 1. Window Setup
    self.title("ECHO AGI // MONITORING STATION")
    self.geometry("1100x700")
    self.grid_columnconfigure(1, weight=1)
    self.grid_rowconfigure(0, weight=1)

    # 2. Data Link
    self.system = None
    self.msg_queue = queue.Queue()
    self.running = False

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
    self.transcript_label.pack(pady=(10,0), anchor="w")

    self.console = ctk.CTkTextbox(self.tab_monitor, width=800, height=400, font=("Consolas", 14))
    self.console.pack(pady=10, padx=10, fill="both", expand=True)
    self.console.insert("0.0", "[SYSTEM READY] Awaiting audio input...\n")
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

def toggle_system(self):
    if not self.running:
        # Start System
        self.running = True
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
        self.running = False
        self.start_btn.configure(text="INITIALIZE SYSTEM", fg_color="green")
        self.status_label.configure(text="STATUS: STOPPED", text_color="gray")

def run_echo_thread(self):
    """Runs the Echo Loop and captures output to GUI."""
    try:
        self.system = EchoSystem()
        # We monkey-patch the print function OR the system's notify method to pipe data here
        # For this example, we rely on the GUI pulling data if we modify EchoSystem,
        # or we just simulate the data feed if you run this without the mic connected.

        # Real implementation:
        # self.system.on_transcript = lambda t: self.msg_queue.put(("TRANSCRIPT", t))
        # self.system.on_arousal = lambda a: self.msg_queue.put(("AROUSAL", a))
        # self.system.listen_loop()

        # For immediate gratification/testing UI logic:
        self.console_log("Core Attached. Listening...")
        self.system.listen_loop() 

    except Exception as e:
        self.msg_queue.put(("ERROR", str(e)))

def console_log(self, text):
    self.msg_queue.put(("LOG", text))

def inject_command(self, cmd):
    self.console_log(f"COMMAND INJECTED: {cmd.upper()}")
    # In real integration: self.system.inject_strategy(cmd)

def update_gui(self):
    """Polls the queue for updates from the AI thread."""
    try:
        while True:
            type, data = self.msg_queue.get_nowait()

            if type == "LOG":
                self.console.configure(state="normal")
                self.console.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {data}\n")
                self.console.see("end")
                self.console.configure(state="disabled")

            elif type == "AROUSAL":
                # Update bar color based on stress
                val = float(data)
                self.arousal_bar.set(val / 10.0)
                self.arousal_val.configure(text=f"{val:.1f}")
                if val > 7.0: self.arousal_bar.configure(progress_color="red")
                elif val > 4.0: self.arousal_bar.configure(progress_color="orange")
                else: self.arousal_bar.configure(progress_color="#3b8ed0")

    except queue.Empty:
        pass

    self.after(100, self.update_gui)

