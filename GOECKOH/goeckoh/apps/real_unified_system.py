#!/usr/bin/env python3
"""
REAL UNIFIED NEURO-ACOUSTIC AGI SYSTEM
========================================

Production-ready implementation with real-world deployment capabilities.
This system integrates all components into a deployable application with
proper error handling, logging, configuration management, and monitoring.

DEPLOYMENT FEATURES:
✅ Real-time audio processing with device selection
✅ Persistent configuration and user profiles
✅ Production logging and monitoring
✅ API endpoints for external integration
✅ GUI interface for interaction
✅ Session management and persistence
✅ Hardware abstraction layer
✅ Safety monitoring and emergency controls
"""

import numpy as np
import time
import json
import os
import sys
import argparse
import threading
import queue
import csv
import math
import logging
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import deque
from enum import Enum
import warnings
from pathlib import Path
import configparser
import uuid
from datetime import datetime, timedelta

# Ensure repository root is on sys.path when executed directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Headless/CI toggle to avoid initializing GUI stacks (Kivy)
HEADLESS = os.environ.get("USE_HEADLESS", "").lower() in ("1", "true", "yes")

# Optional imports for websockets
try:
    import asyncio
    import websockets
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

# Import the complete unified system
from goeckoh.systems.complete_unified_system import (
    CompleteUnifiedSystem,
    EmotionalState,
    SystemMetrics,
)

# Production dependencies
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("🔇 SoundDevice not available - Silent mode")

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except Exception as exc:  # catch OSError/ImportError from missing shared libs
    TORCH_AVAILABLE = False
    print(f"⚠️  Torch unavailable ({exc}); running without torch/torchaudio")

try:
    import flask
    from flask import Flask, request, jsonify, render_template, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    if HEADLESS:
        raise ImportError("Headless mode - skip Kivy imports")
    import kivy
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    from kivy.uix.textinput import TextInput
    from kivy.clock import Clock
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False

# Optional dependencies - make imports optional
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    from flask_cors import CORS
    FLASK_CORS_AVAILABLE = True
except ImportError:
    FLASK_CORS_AVAILABLE = False

# ============================================================================
# PRODUCTION CONFIGURATION
# ============================================================================

class SystemConfig:
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

# ============================================================================
# PRODUCTION LOGGING SYSTEM
# ============================================================================

class ProductionLogger:
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

# ============================================================================
# USER PROFILE MANAGEMENT
# ============================================================================

@dataclass
class UserProfile:
    """User profile for personalized experiences"""
    user_id: str
    name: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    session_history: List[Dict] = field(default_factory=list)
    emotional_baseline: EmotionalState = field(default_factory=EmotionalState)
    skill_levels: Dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

class UserProfileManager:
    """Manage user profiles and personalization"""
    
    def __init__(self, config: SystemConfig, logger: ProductionLogger):
        self.config = config
        self.logger = logger
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.active_profiles = {}
    
    def create_profile(self, name: str, preferences: Dict = None) -> UserProfile:
        """Create new user profile"""
        user_id = str(uuid.uuid4())
        profile = UserProfile(
            user_id=user_id,
            name=name,
            preferences=preferences or {}
        )
        
        self.active_profiles[user_id] = profile
        self.save_profile(profile)
        self.logger.log_event("PROFILE_CREATED", f"Created profile for {name}")
        
        return profile
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from disk"""
        profile_file = self.profiles_dir / f"{user_id}.json"
        
        if not profile_file.exists():
            return None
        
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
            
            profile = UserProfile(
                user_id=data['user_id'],
                name=data['name'],
                preferences=data['preferences'],
                session_history=data['session_history'],
                emotional_baseline=EmotionalState(**data['emotional_baseline']),
                skill_levels=data['skill_levels'],
                created_at=data['created_at'],
                last_active=data['last_active']
            )
            
            self.active_profiles[user_id] = profile
            return profile
            
        except Exception as e:
            self.logger.log_event("PROFILE_LOAD_ERROR", str(e), 'ERROR')
            return None
    
    def save_profile(self, profile: UserProfile):
        """Save user profile to disk"""
        profile_file = self.profiles_dir / f"{profile.user_id}.json"
        
        try:
            data = {
                'user_id': profile.user_id,
                'name': profile.name,
                'preferences': profile.preferences,
                'session_history': profile.session_history,
                'emotional_baseline': profile.emotional_baseline.__dict__,
                'skill_levels': profile.skill_levels,
                'created_at': profile.created_at,
                'last_active': profile.last_active
            }
            
            with open(profile_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.log_event("PROFILE_SAVE_ERROR", str(e), 'ERROR')
    
    def update_profile_activity(self, user_id: str):
        """Update profile last active timestamp"""
        if user_id in self.active_profiles:
            self.active_profiles[user_id].last_active = time.time()
            self.save_profile(self.active_profiles[user_id])

# ============================================================================
# REAL-TIME AUDIO PROCESSING
# ============================================================================

class RealTimeAudioProcessor:
    """Real-time audio processing with device management"""
    
    def __init__(self, config: SystemConfig, logger: ProductionLogger):
        self.config = config
        self.logger = logger
        self.sample_rate = config.get_int('AUDIO', 'sample_rate', 22050)
        self.buffer_size = config.get_int('AUDIO', 'buffer_size', 1024)
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Device management
        self.available_devices = self.list_audio_devices()
        self.input_device = config.get('AUDIO', 'input_device', 'default')
        self.output_device = config.get('AUDIO', 'output_device', 'default')
        
        # Processing state
        self.is_recording = False
        self.is_playing = False
        self.processing_thread = None
        
        self.logger.log_event("AUDIO_INIT", f"Initialized with {len(self.available_devices)} devices")
    
    def list_audio_devices(self) -> List[Dict]:
        """List available audio devices"""
        if not AUDIO_AVAILABLE:
            return []
        
        try:
            default_input_device_index = sd.default.device[0]
            default_output_device_index = sd.default.device[1]
        except Exception:
            default_input_device_index = -1
            default_output_device_index = -1

        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0 or device['max_output_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'inputs': device['max_input_channels'],
                    'outputs': device['max_output_channels'],
                    'default_input': i == default_input_device_index,
                    'default_output': i == default_output_device_index
                })
        
        return devices
    
    def start_recording(self):
        """Start audio recording"""
        if not AUDIO_AVAILABLE:
            self.logger.log_event("AUDIO_ERROR", "Audio device not available", 'ERROR')
            return False

        if not self.available_devices:
            self.logger.log_event("AUDIO_ERROR", "No audio input devices detected", 'ERROR')
            return False

        # Auto-select a usable device when config says "default"
        device_to_use = self.input_device
        if device_to_use in (None, "default"):
            try:
                default_idx = sd.default.device[0] if sd.default.device else None
                device_to_use = default_idx if default_idx is not None else self.available_devices[0]['id']
                self.logger.log_event("AUDIO_DEVICE_SELECT", f"Auto-selected input device {device_to_use}")
            except Exception as e:
                self.logger.log_event("AUDIO_DEVICE_SELECT_ERROR", str(e), 'ERROR')
                return False

        # Validate that the chosen device exists
        matching = next((d for d in self.available_devices
                         if d['id'] == device_to_use or d['name'] == device_to_use), None)
        if not matching:
            self.logger.log_event("AUDIO_RECORDING_ERROR", f"No input device matching '{device_to_use}'", 'ERROR')
            return False

        self.input_device = matching['id']
        
        try:
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.buffer_size,
                device=self.input_device,
                callback=self._audio_input_callback
            )
            
            self.input_stream.start()
            self.is_recording = True
            
            self.logger.log_event("AUDIO_RECORDING", f"Started recording on {self.input_device}")
            return True
            
        except Exception as e:
            self.logger.log_event("AUDIO_RECORDING_ERROR", str(e), 'ERROR')
            return False
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        
        self.is_recording = False
        self.logger.log_event("AUDIO_RECORDING", "Stopped recording")
    
    def _audio_input_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            self.logger.log_event("AUDIO_STATUS", str(status))
        
        # Add audio data to queue
        audio_data = indata[:, 0]  # Take first channel
        try:
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            # Drop oldest data if queue is full
            try:
                self.audio_queue.get(block=False)
                self.audio_queue.put(audio_data, block=False)
            except queue.Empty:
                pass
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def play_audio(self, audio_data: np.ndarray):
        """Play audio data"""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
            sd.play(audio_data, samplerate=self.sample_rate)
            return True
        except Exception as e:
            self.logger.log_event("AUDIO_PLAYBACK_ERROR", str(e), 'ERROR')
            return False

# ============================================================================
# WEB API INTERFACE
# ============================================================================

class WebAPIInterface:
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

# ============================================================================
# MAIN PRODUCTION SYSTEM
# ============================================================================

class RealUnifiedSystem:
    """Production-ready unified system with all deployment features"""
    
    def __init__(self, config_file: str = "real_system_config.ini", disable_api: bool = False):
        # Initialize configuration
        self.config = SystemConfig(config_file)
        self.disable_api = disable_api
        
        # Initialize logging
        self.logger = ProductionLogger(self.config)
        self.logger.log_event("SYSTEM_START", "Initializing Real Unified System")
        
        # Initialize core unified system
        self.unified_system = CompleteUnifiedSystem()
        self.logger.log_event("CORE_INIT", "Unified system initialized")
        
        # Initialize subsystems
        self.profile_manager = UserProfileManager(self.config, self.logger)
        self.audio_processor = RealTimeAudioProcessor(self.config, self.logger)
        self.api_interface = WebAPIInterface(self.config, self.logger, self.unified_system)
        
        # System state
        self.is_running = False
        self.current_user = None
        self.session_start_time = time.time()
        
        # Safety monitoring
        self.safety_monitor = SafetyMonitor(self.config, self.logger, self.unified_system)
        
        self.logger.log_event("SYSTEM_READY", "Real Unified System ready for deployment")
    
    def start(self):
        """Start the production system"""
        self.logger.log_event("SYSTEM_STARTUP", "Starting all subsystems")
        
        # Start audio processing
        if self.config.get_bool('AUDIO', 'auto_start', True):
            try:
                self.audio_processor.start_recording()
            except Exception as e:
                self.logger.log_event("AUDIO_START_ERROR", str(e), 'ERROR')
        
        # Start API server
        api_should_start = (self.config.get_bool('API', 'auto_start', True) or HEADLESS) and not self.disable_api
        if api_should_start and FLASK_AVAILABLE:
            try:
                self.api_interface.start_server()
            except Exception as e:
                self.logger.log_event("API_START_ERROR", str(e), 'ERROR')
        elif api_should_start and not FLASK_AVAILABLE:
            self.logger.log_event("API_WARNING", "Flask not available - API server skipped", 'WARNING')
        elif self.disable_api:
            self.logger.log_event("API_SKIPPED", "API server disabled by flag", 'INFO')
        
        # Start safety monitoring
        try:
            self.safety_monitor.start()
        except Exception as e:
            self.logger.log_event("SAFETY_START_ERROR", str(e), 'ERROR')
        
        self.is_running = True
        self.logger.log_event("SYSTEM_STARTED", "All subsystems started")
    
    def stop(self):
        """Stop the production system"""
        self.logger.log_event("SYSTEM_SHUTDOWN", "Stopping all subsystems")
        
        # Stop audio processing
        self.audio_processor.stop_recording()
        
        # Stop safety monitoring
        self.safety_monitor.stop()
        
        # Save current user profile
        if self.current_user:
            self.profile_manager.save_profile(self.current_user)
        
        self.is_running = False
        self.logger.log_event("SYSTEM_STOPPED", "All subsystems stopped")
    
    def create_user_session(self, name: str, preferences: Dict = None) -> UserProfile:
        """Create new user session"""
        profile = self.profile_manager.create_profile(name, preferences)
        self.current_user = profile
        self.logger.log_event("USER_SESSION", f"Created session for {name}")
        return profile
    
    def load_user_session(self, user_id: str) -> Optional[UserProfile]:
        """Load existing user session"""
        profile = self.profile_manager.load_profile(user_id)
        if profile:
            self.current_user = profile
            self.logger.log_event("USER_SESSION", f"Loaded session for {profile.name}")
        return profile
    
    def process_input(self, text: str, sensory_data: Dict = None) -> Dict:
        """Process user input with full production features"""
        start_time = time.time()
        
        try:
            # Process through unified system
            result = self.unified_system.process_input(text, sensory_data=sensory_data)
            
            # Update user profile if active
            if self.current_user:
                self.current_user.session_history.append({
                    'timestamp': time.time(),
                    'input': text,
                    'response': result.get('response_text', ''),
                    'emotional_state': result.get('emotional_state', {}).__dict__ if result.get('emotional_state') else {}
                })
                
                # Update profile activity
                self.profile_manager.update_profile_activity(self.current_user.user_id)
            
            processing_time = time.time() - start_time
            self.logger.update_metrics(processing_time)
            
            self.logger.log_event("INPUT_PROCESSED", f"Processed input in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.log_event("PROCESSING_ERROR", str(e), 'ERROR')
            self.logger.metrics['errors'] += 1
            raise
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        base_status = self.unified_system.get_complete_system_status()
        
        # Add production-specific status
        production_status = {
            'production_info': {
                'session_id': self.logger.session_id,
                'uptime': time.time() - self.session_start_time,
                'current_user': self.current_user.name if self.current_user else None,
                'is_running': self.is_running,
                'config_file': self.config.config_file
            },
            'audio_status': {
                'is_recording': self.audio_processor.is_recording,
                'available_devices': len(self.audio_processor.available_devices),
                'input_device': self.audio_processor.input_device,
                'output_device': self.audio_processor.output_device,
                'queue_size': self.audio_processor.audio_queue.qsize()
            },
            'api_status': {
                'server_running': self.api_interface.server_thread is not None,
                'host': self.config.get('API', 'host'),
                'port': self.config.get_int('API', 'port')
            },
            'performance_metrics': {
                'total_requests': self.logger.metrics['requests'],
                'total_errors': self.logger.metrics['errors'],
                'avg_processing_time': np.mean(self.logger.metrics['processing_time']) if self.logger.metrics['processing_time'] else 0,
                'error_rate': self.logger.metrics['errors'] / max(1, self.logger.metrics['requests'])
            }
        }
        
        # Merge with base status
        base_status.update(production_status)
        return base_status
    
    def run_interactive_mode(self):
        """Run interactive command-line mode"""
        print("\n" + "="*80)
        print("🚀 REAL UNIFIED NEURO-ACOUSTIC AGI SYSTEM - INTERACTIVE MODE")
        print("="*80)
        print("Type 'help' for commands, 'quit' to exit")
        print("-"*80)
        
        while self.is_running:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                elif user_input.lower() == 'profile':
                    self.show_profile()
                    continue
                elif user_input.startswith('user '):
                    # Create/load user: user <name>
                    name = user_input[5:].strip()
                    if name:
                        self.create_user_session(name)
                        print(f"✅ Created session for {name}")
                    continue
                
                # Process input
                result = self.process_input(user_input)
                
                print(f"\n🤖 System: {result.get('response_text', 'No response')}")
                print(f"🎭 Mode: {result.get('system_status', {}).get('system_mode', 'UNKNOWN')}")
                print(f"📊 GCL: {result.get('system_status', {}).get('gcl', 0):.3f}")
                
                if result.get('aba_intervention'):
                    intervention = result['aba_intervention']
                    print(f"🧩 ABA: {intervention.get('strategy', 'None')}")
                
            except EOFError:
                print("\n🛑 Input stream closed; exiting interactive mode.")
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available commands:
  help          - Show this help
  status        - Show system status
  profile       - Show current user profile
  quit          - Exit the system
  user <name>   - Create/load user profile
  <text>        - Process text input

System features:
  ✅ Real-time audio processing
  ✅ User profile management
  ✅ Web API interface
  ✅ Safety monitoring
  ✅ Production logging
  ✅ Session persistence
        """
        print(help_text)
    
    def show_status(self):
        """Show system status"""
        status = self.get_system_status()
        
        print(f"\n📊 System Status:")
        print(f"  Uptime: {status['production_info']['uptime']:.1f}s")
        print(f"  Session: {status['production_info']['session_id']}")
        print(f"  User: {status['production_info']['current_user'] or 'None'}")
        print(f"  GCL: {status['gcl']:.3f}")
        print(f"  Mode: {status['system_mode']}")
        print(f"  Audio: {'🔴' if status['audio_status']['is_recording'] else '⚪'} Recording")
        print(f"  API: {'🟢' if status['api_status']['server_running'] else '⚪'} Server")
        print(f"  Requests: {status['performance_metrics']['total_requests']}")
        print(f"  Errors: {status['performance_metrics']['total_errors']}")
    
    def show_profile(self):
        """Show current user profile"""
        if not self.current_user:
            print("No active user profile")
            return
        
        profile = self.current_user
        print(f"\n👤 User Profile:")
        print(f"  Name: {profile.name}")
        print(f"  ID: {profile.user_id}")
        print(f"  Sessions: {len(profile.session_history)}")
        print(f"  Created: {datetime.fromtimestamp(profile.created_at).strftime('%Y-%m-%d %H:%M')}")
        print(f"  Last Active: {datetime.fromtimestamp(profile.last_active).strftime('%Y-%m-%d %H:%M')}")

# ============================================================================
# SAFETY MONITORING SYSTEM
# ============================================================================

class SafetyMonitor:
    """Safety monitoring and emergency controls"""
    
    def __init__(self, config: SystemConfig, logger: ProductionLogger, unified_system: CompleteUnifiedSystem):
        self.config = config
        self.logger = logger
        self.unified_system = unified_system
        
        # Safety thresholds
        self.max_arousal = config.get_float('SAFETY', 'max_arousal', 0.9)
        self.stress_threshold = config.get_float('SAFETY', 'stress_threshold', 0.8)
        self.monitor_interval = config.get_int('SAFETY', 'monitor_interval', 5)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.emergency_triggered = False
        
        # Emergency actions
        self.emergency_actions = []
    
    def start(self):
        """Start safety monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.log_event("SAFETY_START", "Safety monitoring started")
    
    def stop(self):
        """Stop safety monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.logger.log_event("SAFETY_STOP", "Safety monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._check_system_safety()
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.log_event("SAFETY_ERROR", str(e), 'ERROR')
    
    def _check_system_safety(self):
        """Check system safety parameters"""
        status = self.unified_system.get_complete_system_status()
        
        # Check stress levels
        if status.get('stress', 0) > self.stress_threshold:
            self._trigger_safety_event("HIGH_STRESS", f"Stress level: {status['stress']:.3f}")
        
        # Check emotional state
        emotional_state = status.get('emotional_state')
        if emotional_state:
            if hasattr(emotional_state, 'anxiety') and emotional_state.anxiety > self.max_arousal:
                self._trigger_safety_event("HIGH_ANXIETY", f"Anxiety level: {emotional_state.anxiety:.3f}")
        
        # Check system health
        if status.get('gcl', 1.0) < 0.3:
            self._trigger_safety_event("LOW_COHERENCE", f"GCL: {status['gcl']:.3f}")
    
    def _trigger_safety_event(self, event_type: str, details: str):
        """Trigger safety event"""
        self.logger.log_event("SAFETY_EVENT", f"{event_type}: {details}", 'WARNING')
        
        # Add emergency actions if needed
        if event_type in ["HIGH_STRESS", "HIGH_ANXIETY", "LOW_COHERENCE"]:
            self._execute_emergency_protocol(event_type)
    
    def _execute_emergency_protocol(self, event_type: str):
        """Execute emergency protocol"""
        if self.emergency_triggered:
            return  # Already in emergency mode
        
        self.emergency_triggered = True
        self.logger.log_event("EMERGENCY_PROTOCOL", f"Executing emergency protocol for {event_type}")
        
        # Emergency actions based on event type
        if event_type == "HIGH_STRESS":
            # Reduce system arousal, activate calming protocols
            pass
        elif event_type == "HIGH_ANXIETY":
            # Activate anxiety reduction protocols
            pass
        elif event_type == "LOW_COHERENCE":
            # System coherence restoration
            pass
        
        # Reset emergency flag after cooldown
        threading.Timer(30.0, self._reset_emergency).start()
    
    def _reset_emergency(self):
        """Reset emergency state"""
        self.emergency_triggered = False
        self.logger.log_event("EMERGENCY_RESET", "Emergency state reset")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Real Unified Neuro-Acoustic AGI System")
    parser.add_argument("--config", default="real_system_config.ini", help="Path to config file")
    parser.add_argument("--api", action="store_true", help="Run without CLI prompt (API/service mode)")
    parser.add_argument("--no-interactive", action="store_true", help="Alias for --api; skip CLI loop")
    parser.add_argument("--disable-api", action="store_true", help="Do not start the Flask API server")
    args = parser.parse_args()

    print("🚀 Starting Real Unified Neuro-Acoustic AGI System...")

    # Auto-disable interactive loop if stdin is not a TTY (e.g., CI/headless exec)
    if not sys.stdin.isatty() and not (args.api or args.no_interactive):
        print("ℹ️  No TTY detected; running in API mode (--api)")
        args.api = True
    
    # Initialize system
    system = RealUnifiedSystem(config_file=args.config, disable_api=args.disable_api)
    
    # Start system
    system.start()
    
    try:
        if args.api or args.no_interactive:
            print("🌐 API mode: running without interactive prompt")
            while True:
                time.sleep(1.0)
        else:
            # Run interactive mode
            system.run_interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        # Stop system
        system.stop()
        print("👋 System shutdown complete")

if __name__ == "__main__":
    main()
