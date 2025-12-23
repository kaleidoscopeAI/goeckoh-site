"""Production system configuration management"""

def __init__(self, config_file: str = "real_system_config.ini"):
    self.config_file = config_file
    self.config = configparser.ConfigParser()
    self.load_config()

def load_config(self):
    """Load configuration from file or create defaults"""
    if os.path.exists(self.config_file):
        self.config.read(self.config_file)
    else:
        self.create_default_config()
        self.save_config()

def create_default_config(self):
    """Create default production configuration"""
    self.config['SYSTEM'] = {
        'log_level': 'INFO',
        'max_sessions': 100,
        'session_timeout': '3600',
        'backup_interval': '300',
        'auto_save': 'true'
    }

    self.config['AUDIO'] = {
        'sample_rate': '22050',
        'buffer_size': '1024',
        'input_device': 'default',
        'output_device': 'default',
        'channels': '1',
        'latency': 'low'
    }

    self.config['API'] = {
        'host': 'localhost',
        'port': '8080',
        'enable_cors': 'true',
        'rate_limit': '100',
        'auth_required': 'false'
    }

    self.config['GUI'] = {
        'theme': 'dark',
        'window_size': '1200x800',
        'auto_start': 'true',
        'minimize_to_tray': 'true'
    }

    self.config['SAFETY'] = {
        'max_arousal': '0.9',
        'stress_threshold': '0.8',
        'emergency_stop': 'true',
        'monitor_interval': '5'
    }

    self.config['MEMORY'] = {
        'max_memories': '10000',
        'retention_days': '30',
        'compression': 'true',
        'encryption': 'false'
    }

def save_config(self):
    """Save configuration to file"""
    with open(self.config_file, 'w') as f:
        self.config.write(f)

def get(self, section: str, key: str, fallback=None):
    """Get configuration value"""
    return self.config.get(section, key, fallback=fallback)

def get_int(self, section: str, key: str, fallback=0):
    """Get integer configuration value"""
    return self.config.getint(section, key, fallback=fallback)

def get_float(self, section: str, key: str, fallback=0.0):
    """Get float configuration value"""
    return self.config.getfloat(section, key, fallback=fallback)

def get_bool(self, section: str, key: str, fallback=False):
    """Get boolean configuration value"""
    return self.config.getboolean(section, key, fallback=fallback)

