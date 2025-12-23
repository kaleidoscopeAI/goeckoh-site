def __init__(self):
    self.root = tk.Tk()
    self.root.title("Echo v4.0 - First-Time Setup")
    self.root.geometry("600x500")
    self.root.resizable(False, False)

    self.voice_path = None
    self.ollama_enabled = tk.BooleanVar(value=False)

    self.create_ui()

def create_ui(self):
    # Header
    header = tk.Frame(self.root, bg="#4f46e5", height=80)
    header.pack(fill=tk.X)

    title = tk.Label(
        header,
        text="🎙️ Echo v4.0",
        bg="#4f46e5",
        fg="white",
        font=("Helvetica", 24, "bold")
    )
    title.pack(pady=20)

    # Main content
    content = tk.Frame(self.root, padx=30, pady=20)
    content.pack(fill=tk.BOTH, expand=True)

    # Welcome text
    welcome = tk.Label(
        content,
        text="Welcome to Jackson's Crystalline Speech Companion",
        font=("Helvetica", 14),
        wraplength=500
    )
    welcome.pack(pady=(0, 20))

    # Step 1: Voice sample
    step1_frame = tk.LabelFrame(content, text="Step 1: Voice Sample", padx=10, pady=10)
    step1_frame.pack(fill=tk.X, pady=10)

    tk.Label(
        step1_frame,
        text="Record 10-30 seconds of the child speaking naturally.\\n"
             "This will be used to clone their voice.",
        wraplength=500,
        justify=tk.LEFT
    ).pack(anchor=tk.W)

    tk.Button(
        step1_frame,
        text="📁 Select Voice Recording (WAV file)",
        command=self.select_voice_file,
        bg="#4f46e5",
        fg="white",
        font=("Helvetica", 11, "bold"),
        padx=20,
        pady=10
    ).pack(pady=10)

    self.voice_label = tk.Label(step1_frame, text="No file selected", fg="gray")
    self.voice_label.pack()

    # Step 2: LLM (optional)
    step2_frame = tk.LabelFrame(content, text="Step 2: Inner Voice AI (Optional)", padx=10, pady=10)
    step2_frame.pack(fill=tk.X, pady=10)

    tk.Label(
        step2_frame,
        text="Enable local AI for gentle inner voice phrases?\\n"
             "Requires Ollama to be installed separately.",
        wraplength=500,
        justify=tk.LEFT
    ).pack(anchor=tk.W)

    tk.Checkbutton(
        step2_frame,
        text="Enable AI Inner Voice",
        variable=self.ollama_enabled,
        font=("Helvetica", 10)
    ).pack(anchor=tk.W, pady=5)

    # Start button
    tk.Button(
        content,
        text="🚀 Start Echo",
        command=self.start_echo,
        bg="#22c55e",
        fg="white",
        font=("Helvetica", 14, "bold"),
        padx=40,
        pady=15
    ).pack(pady=20)

def select_voice_file(self):
    path = filedialog.askopenfilename(
        title="Select Voice Recording",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    if path:
        self.voice_path = path
        self.voice_label.config(text=f"✓ {Path(path).name}", fg="green")

def start_echo(self):
    if not self.voice_path:
        messagebox.showerror("Error", "Please select a voice recording first!")
        return

    # Copy voice file to config location
    config_dir = Path.home() / ".echo_companion" / "voices"
    config_dir.mkdir(parents=True, exist_ok=True)

    target_path = config_dir / "child_ref.wav"
    shutil.copy2(self.voice_path, target_path)

    # Save config
    config = {
        "voice_sample": str(target_path),
        "llm_enabled": self.ollama_enabled.get(),
        "first_run_complete": True
    }

    config_file = Path.home() / ".echo_companion" / "setup.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    messagebox.showinfo(
        "Setup Complete",
        "Echo is ready!\\n\\nThe application will now start.\\n\\n"
        "Open your browser to: http://localhost:8000/static/index.html"
    )

    self.root.destroy()

    # Start the server
    self.launch_server()

def launch_server(self):
    import subprocess

    # Get the installation directory
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys._MEIPASS)
    else:
        app_dir = Path(__file__).parent

    # Start uvicorn server
    subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ], cwd=app_dir)

def run(self):
    self.root.mainloop()

