class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def print_status(msg, type="INFO"):
    prefix = {
        "INFO": "[*]",
        "OK": f"{Colors.GREEN}[+]{Colors.ENDC}",
        "WARN": f"{Colors.YELLOW}[!]{Colors.ENDC}",
        "ERR": f"{Colors.FAIL}[x]{Colors.ENDC}"
    }.get(type, "[?]")
    print(f"{prefix} {msg}")

def check_rust_toolchain():
    print_status("Checking Rust installation...")
    try:
        subprocess.run(["cargo", "--version"], check=True, stdout=subprocess.PIPE)
        print_status("Rust compiler found.", "OK")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("Rust not found. Install via https://rustup.rs/", "ERR")
        return False

def build_rust_extension():
    """
    Compiles the bio_audio Rust crate into a shared library accessible by Python.
    """
    print_status("Building Physics Kernel (bio_audio)...", "INFO")
    
    os_type = platform.system()
    target_dir = os.path.join(os.getcwd(), "bio_audio", "target", "release")
    root_dir = os.getcwd()
    
    # 1. Run Cargo Build
    try:
        os.chdir("bio_audio")
        cmd = ["cargo", "build", "--release"]
        subprocess.run(cmd, check=True)
        os.chdir(root_dir)
    except subprocess.CalledProcessError:
        print_status("Compilation Failed.", "ERR")
        return False

    # 2. Locate and Move Artifact
    # Windows: .dll -> .pyd
    # Linux/Mac: .so / .dylib -> .so
    
    src_name = ""
    dest_name = ""
    
    if os_type == "Windows":
        src_name = "bio_audio.dll"
        dest_name = "bio_audio.pyd"
    elif os_type == "Darwin": # Mac
        src_name = "libbio_audio.dylib"
        dest_name = "bio_audio.so"
    else: # Linux
        src_name = "libbio_audio.so"
        dest_name = "bio_audio.so"
        
    src_path = os.path.join(target_dir, src_name)
    dest_path = os.path.join(root_dir, dest_name)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        print_status(f"Kernel linked: {dest_name}", "OK")
        return True
    else:
        print_status(f"Artifact not found at {src_path}", "ERR")
        return False

def main():
    print(f"{Colors.CYAN}=== NEURO-ACOUSTIC EXOCORTEX LAUNCHER ==={Colors.ENDC}")
    
    # 1. Environment Check
    if not check_rust_toolchain():
        print_status("Cannot build kernel. Continuing with mocks is possible but not recommended.", "WARN")
        time.sleep(2)
    else:
        # 2. Build Rust Kernel
        if not os.path.exists("bio_audio.so") and not os.path.exists("bio_audio.pyd"):
            success = build_rust_extension()
            if not success:
                print_status("Using Python-only mode.", "WARN")

    # 3. Launch GUI
    print_status("Initializing Crystalline Heart...", "INFO")
    try:
        subprocess.run([sys.executable, "gui_main.py"])
    except KeyboardInterrupt:
        print_status("Shutdown complete.", "OK")

