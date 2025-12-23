"""Manages system components lifecycle"""

def __init__(self, config_path: str):
    """Initialize with configuration"""
    with open(config_path, 'r') as f:
        self.config = json.load(f)

    self.processes = {}
    self.activated_env = False

def _activate_env(self):
    """Activate the virtual environment for subprocess calls"""
    if self.activated_env:
        return

    # Get the activate script path
    if sys.platform == 'win32':
        activate_script = Path.cwd() / "venv" / "Scripts" / "activate.bat"
        self.activate_cmd = f'"{activate_script}"'
    else:
        activate_script = Path.cwd() / "venv" / "bin" / "activate"
        self.activate_cmd = f'source "{activate_script}"'

    self.activated_env = True

def start_component(self, name: str, script_path: str, args: List[str] = None):
    """Start a system component as a subprocess"""
    self._activate_env()

    if name in self.processes and self.processes[name].poll() is None:
        logger.info(f"Component {name} is already running")
        return

    args = args or []
    cmd = f'{self.activate_cmd} && python "{script_path}" {" ".join(args)}'

    logger.info(f"Starting component: {name}")
    if sys.platform == 'win32':
        process = subprocess.Popen(cmd, shell=True)
    else:
        process = subprocess.Popen(cmd, shell=True, executable="/bin/bash")

    self.processes[name] = process
    logger.info(f"Started {name} (PID: {process.pid})")

def stop_component(self, name: str):
    """Stop a running component"""
    if name not in self.processes:
        logger.warning(f"Component {name} is not running")
        return

    process = self.processes[name]
    if process.poll() is None:
        logger.info(f"Stopping component: {name}")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Component {name} did not terminate gracefully, killing it")
            process.kill()

    del self.processes[name]

def start_api_server(self):
    """Start the FastAPI server"""
    self._activate_env()

    host = self.config.get("host", "0.0.0.0")
    port = self.config.get("port", 8000)

    cmd = f'{self.activate_cmd} && python -m uvicorn src.main:app --host={host} --port={port} --reload'

    logger.info(f"Starting API server on {host}:{port}")
    if sys.platform == 'win32':
        process = subprocess.Popen(cmd, shell=True)
    else:
        process = subprocess.Popen(cmd, shell=True, executable="/bin/bash")

    self.processes["api_server"] = process
    logger.info(f"Started API server (PID: {process.pid})")

def stop_all(self):
    """Stop all running components"""
    for name in list(self.processes.keys()):
        self.stop_component(name)

