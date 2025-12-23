import os, sys, subprocess, shutil, platform

def main():
    print("Checking Physics Kernel...")
    os_type = platform.system()
    ext = ".pyd" if os_type == "Windows" else ".so"
    
    if not os.path.exists(f"bio_audio{ext}"):
        print("Compiling Rust...")
        os.chdir("bio_audio")
        subprocess.run(["cargo", "build", "--release"], check=True)
        os.chdir("..")
        
        src = f"bio_audio/target/release/libbio_audio{'.dylib' if os_type=='Darwin' else '.so'}"
        if os_type == "Windows": src = "bio_audio/target/release/bio_audio.dll"
        shutil.copy(src, f"./bio_audio{ext}")

    print("Launching Exocortex...")
    subprocess.run([sys.executable, "gui_main.py"])

