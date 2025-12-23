"""Enhanced logging system for production deployment"""

def __init__(self, config: SystemConfig):
    self.config = config
    self.setup_logging()
    self.session_id = str(uuid.uuid4())
    self.start_time = time.time()

    # Performance metrics
    self.metrics = {
        'requests': 0,
        'errors': 0,
        'processing_time': deque(maxlen=1000),
        'memory_usage': deque(maxlen=100),
        'cpu_usage': deque(maxlen=100)
    }

def setup_logging(self):
    """Setup production logging with rotation and formatting"""
    log_level = self.config.get('SYSTEM', 'log_level', 'INFO')

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Setup file logging
    log_file = log_dir / f"real_system_{datetime.now().strftime('%Y%m%d')}.log"
    try:
        file_handler = logging.FileHandler(log_file)
    except PermissionError:
        fallback_file = log_dir / f"real_system_{datetime.now().strftime('%Y%m%d')}_{os.getpid()}.log"
        print(f"⚠️  Cannot write to {log_file}, using {fallback_file}")
        file_handler = logging.FileHandler(fallback_file)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            logging.StreamHandler(sys.stdout)
        ]
    )

    self.logger = logging.getLogger('RealUnifiedSystem')

    # Add performance metrics handler
    metrics_path = log_dir / "metrics.log"
    try:
        self.metrics_handler = logging.FileHandler(metrics_path)
    except PermissionError:
        fallback_metrics = log_dir / f"metrics_{os.getpid()}.log"
        print(f"⚠️  Cannot write to {metrics_path}, using {fallback_metrics}")
        self.metrics_handler = logging.FileHandler(fallback_metrics)
    self.metrics_handler.setFormatter(
        logging.Formatter('%(asctime)s - METRICS - %(message)s')
    )
    self.metrics_logger = logging.getLogger('Metrics')
    self.metrics_logger.addHandler(self.metrics_handler)
    self.metrics_logger.setLevel(logging.INFO)

def log_event(self, event_type: str, message: str, level: str = 'INFO'):
    """Log system event with context"""
    log_message = f"[{self.session_id}] {event_type}: {message}"
    getattr(self.logger, level.lower())(log_message)

def log_metrics(self):
    """Log performance metrics"""
    if self.metrics['processing_time']:
        avg_time = np.mean(self.metrics['processing_time'])
        self.metrics_logger.info(f"AVG_PROCESSING_TIME:{avg_time:.3f}")

    if self.metrics['memory_usage']:
        avg_memory = np.mean(self.metrics['memory_usage'])
        self.metrics_logger.info(f"AVG_MEMORY_USAGE:{avg_memory:.2f}")

    self.metrics_logger.info(f"TOTAL_REQUESTS:{self.metrics['requests']}")
    self.metrics_logger.info(f"TOTAL_ERRORS:{self.metrics['errors']}")

def update_metrics(self, processing_time: float = None, memory_usage: float = None):
    """Update performance metrics"""
    if processing_time is not None:
        self.metrics['processing_time'].append(processing_time)

    if memory_usage is not None:
        self.metrics['memory_usage'].append(memory_usage)

    self.metrics['requests'] += 1

