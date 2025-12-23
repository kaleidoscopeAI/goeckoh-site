def setup_virtual_environment():
    venv_dir = Path(".venv")
    if sys.prefix != sys.base_prefix:
        print("Already in a virtual environment.")
        return True
    if not venv_dir.exists():
        print(f"Creating virtual environment in {venv_dir}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to create virtual environment: {e}")
            return False
    python_exec = venv_dir / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    if not python_exec.exists():
        print(f"Python executable not found at {python_exec}")
        return False
    print("Installing required dependencies...")
    requirements = ["numpy", "networkx", "scipy", "matplotlib", "tqdm", "psutil", "colorama"]
    if TIMESCALE_ENABLED:
        requirements.append("psycopg2-binary")
    try:
        subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_exec), "-m", "pip", "install"] + requirements, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False
    if sys.executable != str(python_exec):
        print(f"Relaunching with virtual environment Python: {python_exec}")
        os.execv(str(python_exec), [str(python_exec)] + sys.argv)
    return True

