"""REST API interface for external integration"""

def __init__(self, config: SystemConfig, logger: ProductionLogger, unified_system: CompleteUnifiedSystem):
    self.config = config
    self.logger = logger
    self.unified_system = unified_system
    self.app = None
    self.server_thread = None

    if FLASK_AVAILABLE:
        self.setup_flask_app()

def setup_flask_app(self):
    """Setup Flask application"""
    self.app = Flask(__name__)
    self.app.config['JSON_SORT_KEYS'] = False

    # CORS setup
    if self.config.get_bool('API', 'enable_cors', True) and FLASK_CORS_AVAILABLE:
        from flask_cors import CORS
        CORS(self.app)

    # Routes
    self.app.route('/health', methods=['GET'])(self.health_check)
    self.app.route('/status', methods=['GET'])(self.system_status)
    self.app.route('/process', methods=['POST'])(self.process_input)
    self.app.route('/audio/start', methods=['POST'])(self.start_audio)
    self.app.route('/audio/stop', methods=['POST'])(self.stop_audio)
    self.app.route('/profiles', methods=['GET', 'POST'])(self.manage_profiles)
    self.app.route('/metrics', methods=['GET'])(self.get_metrics)

    self.logger.log_event("API_INIT", "Flask API routes configured")

def health_check(self):
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0',
        'components': {
            'unified_system': True,
            'audio': AUDIO_AVAILABLE,
            'flask': FLASK_AVAILABLE
        }
    })

def system_status(self):
    """System status endpoint"""
    status = self.unified_system.get_complete_system_status()
    return jsonify(status)

def process_input(self):
    """Process input endpoint"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text input'}), 400

        text = data['text']
        sensory_data = data.get('sensory_data', {})

        start_time = time.time()
        result = self.unified_system.process_input(text, sensory_data=sensory_data)
        processing_time = time.time() - start_time

        # Update metrics
        self.logger.update_metrics(processing_time)

        return jsonify({
            'success': True,
            'result': result,
            'processing_time': processing_time
        })

    except Exception as e:
        self.logger.log_event("API_ERROR", str(e), 'ERROR')
        return jsonify({'error': str(e)}), 500

def start_audio(self):
    """Start audio processing endpoint"""
    # Implementation would depend on audio processor
    return jsonify({'status': 'audio_started'})

def stop_audio(self):
    """Stop audio processing endpoint"""
    # Implementation would depend on audio processor
    return jsonify({'status': 'audio_stopped'})

def manage_profiles(self):
    """Profile management endpoint"""
    if request.method == 'GET':
        # List profiles
        return jsonify({'profiles': []})  # Implementation needed
    elif request.method == 'POST':
        # Create profile
        return jsonify({'status': 'profile_created'})  # Implementation needed

def get_metrics(self):
    """Get system metrics"""
    return jsonify({
        'timestamp': time.time(),
        'uptime': time.time() - self.logger.start_time,
        'requests': self.logger.metrics['requests'],
        'errors': self.logger.metrics['errors'],
        'avg_processing_time': np.mean(self.logger.metrics['processing_time']) if self.logger.metrics['processing_time'] else 0
    })

def start_server(self):
    """Start the API server"""
    if not FLASK_AVAILABLE:
        self.logger.log_event("API_WARNING", "Flask not available - API server not started", 'WARNING')
        return False

    host = self.config.get('API', 'host', 'localhost')
    port = self.config.get_int('API', 'port', 8080)
    # In containers/headless environments bind on all interfaces
    if HEADLESS or host in ("localhost", "127.0.0.1"):
        host = "0.0.0.0"

    def run_server():
        try:
            self.app.run(host=host, port=port, debug=False)
        except Exception as exc:
            self.logger.log_event("API_ERROR", f"Failed to start API server: {exc}", 'ERROR')
            print(f"⚠️  API server failed to start: {exc}")

    self.server_thread = threading.Thread(target=run_server, daemon=True)
    self.server_thread.start()

    self.logger.log_event("API_START", f"Server started on {host}:{port}")
    return True

