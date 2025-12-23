def ensure_venv():
    if os.environ.get("AGI_BOOTED") != "1":
        if not VENV.exists():
            venv.create(VENV, with_pip=True)
        pip = str(VENV / "bin/pip")
        subprocess.run([pip, "install"] + REQ)
        env = os.environ.copy()
        env["AGI_BOOTED"] = "1"
        os.execve(str(VENV / "bin/python"), [str(VENV / "bin/python"), __file__], env)

