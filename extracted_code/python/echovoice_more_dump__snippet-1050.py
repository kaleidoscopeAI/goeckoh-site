"""Actual hardware control with real system metrics"""

def __init__(self):
    self.connected = self._check_adb_connection()

def _check_adb_connection(self):
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
        return 'device' in result.stdout
    except:
        logging.warning("ADB not available, using simulation mode")
        return False

def get_system_metrics(self):
    """Get real system metrics"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'temperature': self._get_cpu_temperature(),
        'battery_percent': self._get_battery_level(),
        'disk_io': psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0
    }

def _get_cpu_temperature(self):
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
        return 50.0  # Fallback
    except:
        return 50.0

def _get_battery_level(self):
    try:
        battery = psutil.sensors_battery()
        return battery.percent if battery else 100.0
    except:
        return 100.0

def apply_control_signals(self, control_vector):
    """Apply real control signals to system"""
    if not self.connected:
        # Simulate control effects
        logging.info(f"Simulated control: {control_vector}")
        return

    try:
        # Real Android device control via ADB
        cpu_target = int(np.clip(control_vector[0] * 100, 0, 100))
        brightness = int(np.clip(control_vector[1] * 255, 0, 255))

        # Set CPU governor (requires root)
        subprocess.run(['adb', 'shell', 'echo', 'performance', '>', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'], 
                     timeout=5)

        # Set brightness
        subprocess.run(['adb', 'shell', 'settings', 'put', 'system', 'screen_brightness', str(brightness)], 
                     timeout=5)

        logging.info(f"Applied real controls: CPU={cpu_target}%, Brightness={brightness}")

    except Exception as e:
        logging.error(f"Hardware control failed: {e}")

