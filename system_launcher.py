#!/usr/bin/env python3
"""
Bubble System Unified Launcher - REVOLUTIONARY INTEGRATION
=========================================================

Groundbreaking therapeutic system combining:
- Voice therapy & speech correction
- 3D AI-powered universe creation (Cognitive Nebula)
- Real-time AI image generation from voice
- Therapeutic gaming mechanics
- One-click life-changing experience

This system transforms speech therapy into creative universe exploration,
where every word creates AI-generated images in 3D space.
"""

import os
import sys
import argparse
import logging
import threading
import time
import signal
import subprocess
import asyncio
import queue
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class SystemConfig:
    """System configuration with validation"""
    mode: str  # clinician, child, universe, game, pipeline
    profile_name: Optional[str] = None
    voice_profile_path: Optional[str] = None
    log_level: str = "INFO"
    enable_visualization: bool = True
    enable_clinical_logging: bool = True
    auto_download_models: bool = True
    enable_cognitive_nebula: bool = True
    enable_ai_image_generation: bool = True
    universe_theme: str = "therapeutic"
    ai_model: str = "stable_diffusion"

class SystemHealthChecker:
    """Comprehensive system health validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.issues = []
        
    def check_python_modules(self) -> bool:
        """Check required Python packages"""
        required = [
            'numpy', 'scipy', 'sounddevice', 'sherpa_onnx',
            'kivy', 'textual', 'requests'
        ]
        
        missing = []
        for module in required:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            self.issues.append(f"Missing packages: {missing}")
            return False
        return True
    
    def check_audio_hardware(self) -> bool:
        """Test audio hardware availability"""
        try:
            import sounddevice as sd  # type: ignore
            devices = sd.query_devices()
            if not devices:
                self.issues.append("No audio devices found")
                return False
            return True
        except Exception as e:
            self.issues.append(f"Audio hardware check failed: {e}")
            return False
    
    def check_model_files(self) -> bool:
        """Verify neural network models are present"""
        model_paths = [
            "assets/model_stt/tokens.txt",
            "assets/model_stt/encoder-epoch-99-avg-1.onnx", 
            "assets/model_stt/decoder-epoch-99-avg-1.onnx",
            "assets/model_tts/en_US-lessac-medium.onnx"
        ]
        
        missing = []
        for path in model_paths:
            if not (PROJECT_ROOT / path).exists():
                missing.append(path)
        
        if missing:
            self.issues.append(f"Missing model files: {missing}")
            return False
        return True
    
    def check_configuration(self) -> bool:
        """Validate system configuration"""
        config_file = PROJECT_ROOT / "config.yaml"
        if not config_file.exists():
            self.issues.append("config.yaml not found")
            return False
        
        # Additional config validation can go here
        return True
    
    def run_full_check(self) -> bool:
        """Run all health checks"""
        self.logger.info("Running system health checks...")
        
        checks = [
            self.check_python_modules,
            self.check_audio_hardware, 
            self.check_model_files,
            self.check_configuration
        ]
        
        all_passed = True
        for check in checks:
            if not check():
                all_passed = False
        
        if self.issues:
            self.logger.error("Health issues found:")
            for issue in self.issues:
                self.logger.error(f"  - {issue}")
        else:
            self.logger.info("All health checks passed!")
        
        return all_passed

class DesktopAudioBridge:
    """Desktop audio processing implementation"""
    
    def __init__(self, mic_q, spk_q, sr: int):
        self.mic_q = mic_q
        self.spk_q = spk_q
        self.sr = sr
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start desktop audio processing"""
        import sounddevice as sd  # type: ignore
        import numpy as np
        
        def audio_callback(indata, outdata, frames, time, status):
            """Real-time audio processing callback"""
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            
            # Input processing
            if len(indata) > 0:
                audio_data = indata.flatten().astype(np.float32)
                self.mic_q.put(audio_data)
            
            # Output processing  
            try:
                output_data = self.spk_q.get_nowait()
                outdata[:] = output_data.reshape(outdata.shape)
            except:
                outdata[:] = 0
        
        try:
            self.stream = sd.Stream(
                samplerate=self.sr,
                channels=1,
                dtype=np.float32,
                callback=audio_callback
            )
            self.stream.start()
            self.running = True
            self.logger.info("Desktop audio bridge started")
        except Exception as e:
            self.logger.error(f"Failed to start audio bridge: {e}")
            raise
    
    def stop(self):
        """Stop desktop audio processing"""
        if hasattr(self, 'stream') and self.running:
            self.stream.stop()
            self.running = False
            self.logger.info("Desktop audio bridge stopped")

class SystemOrchestrator:
    """Main system orchestrator"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.health_checker = SystemHealthChecker()
        self.components = {}
        self.running = False
        self.nebula_process = None
        self.ai_generator = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize_audio_system(self):
        """Initialize audio processing components"""
        self.logger.info("Initializing audio system...")
        
        # Create audio queues
        mic_q = queue.Queue()
        spk_q = queue.Queue()
        
        # Initialize appropriate audio bridge
        try:
            from platform_utils import is_android
            if is_android():
                from audio_mobile import AndroidAudioBridge
                audio_bridge = AndroidAudioBridge(mic_q, spk_q, 16000)
            else:
                audio_bridge = DesktopAudioBridge(mic_q, spk_q, 16000)
            
            self.components['audio'] = {
                'bridge': audio_bridge,
                'mic_queue': mic_q,
                'spk_queue': spk_q
            }
            
            self.logger.info("Audio system initialized")
            
        except Exception as e:
            self.logger.error(f"Audio initialization failed: {e}")
            raise
    
    def initialize_neuro_backend(self):
        """Initialize neural processing backend"""
        self.logger.info("Initializing neuro backend...")
        
        try:
            from neuro_backend import NeuroKernel
            
            # Get audio queues
            audio_queues = self.components.get('audio', {})
            mic_q = audio_queues.get('mic_queue')
            
            # Create neuro kernel with UI queue
            ui_queue = queue.Queue()
            neuro_kernel = NeuroKernel(ui_queue=ui_queue)
            
            self.components['neuro'] = {
                'kernel': neuro_kernel,
                'ui_queue': ui_queue
            }
            
            self.logger.info("Neuro backend initialized")
            
        except Exception as e:
            self.logger.error("Neuro backend initialization failed: %s", e)
            raise
    
    def initialize_ui_system(self):
        """Initialize user interface based on mode"""
        self.logger.info(f"Initializing UI for mode: {self.config.mode}")
        
        try:
            if self.config.mode == "clinician":
                # Launch clinician dashboard
                from main_app import run_clinician_dashboard
                ui_queue = self.components['neuro']['ui_queue']
                
                def run_ui():
                    run_clinician_dashboard(ui_queue)
                
                ui_thread = threading.Thread(target=run_ui, daemon=True)
                self.components['ui'] = {'thread': ui_thread}
                
            elif self.config.mode == "child":
                # Launch child interface
                from main_app import ChildUI
                ui_queue = self.components['neuro']['ui_queue']
                
                def run_ui():
                    child_ui = ChildUI(ui_queue)
                    child_ui.run()
                
                ui_thread = threading.Thread(target=run_ui, daemon=True)
                self.components['ui'] = {'thread': ui_thread}
                
            elif self.config.mode == "universe":
                # Launch Cognitive Nebula universe mode
                self.initialize_cognitive_nebula()
                
            elif self.config.mode == "game":
                # Launch therapeutic game mode
                self.initialize_therapeutic_game()
                
            else:
                self.logger.warning(f"Unknown UI mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error("UI initialization failed: %s", e)
            raise
    
    def initialize_cognitive_nebula(self):
        """Initialize Cognitive Nebula 3D universe"""
        if not self.config.enable_cognitive_nebula:
            self.logger.warning("Cognitive Nebula disabled in config")
            return
        
        try:
            self.logger.info("Initializing Cognitive Nebula 3D universe...")
            
            nebula_path = PROJECT_ROOT / "cognitive-nebula"
            if not nebula_path.exists():
                self.logger.error(f"Cognitive Nebula not found at {nebula_path}")
                return
            
            # Start Cognitive Nebula as subprocess
            nebula_cmd = ["npm", "run", "dev"]
            self.nebula_process = subprocess.Popen(
                nebula_cmd,
                cwd=nebula_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.components['cognitive_nebula'] = {
                'process': self.nebula_process,
                'url': 'http://localhost:5173'
            }
            
            # Initialize AI image generation if enabled
            if self.config.enable_ai_image_generation:
                self.initialize_ai_generation()
            
            self.logger.info("Cognitive Nebula initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Cognitive Nebula initialization failed: {e}")
            raise
    
    def initialize_ai_generation(self):
        """Initialize AI image generation system"""
        try:
            self.logger.info("Initializing AI image generation...")
            
            # Import AI generation system
            sys.path.insert(0, str(PROJECT_ROOT / "integrations"))
            from ai_image_generation_system import CognitiveNebulaAIIntegration  # type: ignore
            
            # Create AI integration instance
            self.ai_generator = CognitiveNebulaAIIntegration()
            
            # Initialize asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.ai_generator.initialize())
            loop.close()
            
            self.components['ai_generation'] = {
                'generator': self.ai_generator
            }
            
            self.logger.info("AI image generation initialized")
            
        except Exception as e:
            self.logger.error(f"AI generation initialization failed: {e}")
            # Continue without AI generation
            self.logger.warning("Continuing without AI image generation")
    
    def initialize_therapeutic_game(self):
        """Initialize therapeutic game mode"""
        try:
            self.logger.info("Initializing therapeutic game mode...")
            
            # Initialize Cognitive Nebula first
            self.initialize_cognitive_nebula()
            
            # Import game system
            sys.path.insert(0, str(PROJECT_ROOT / "integrations"))
            from voice_universe_game import VoiceUniverseGame, GameUIController  # type: ignore
            
            # Create game engine
            if self.ai_generator:
                from cognitive_nebula_integration import CognitiveNebulaInterface  # type: ignore
                nebula_interface = CognitiveNebulaInterface(self)
                game_engine = VoiceUniverseGame(nebula_interface)
            else:
                self.logger.warning("Game mode requires AI generation")
                return
            
            # Create UI controller
            ui_controller = GameUIController(game_engine)
            
            self.components['therapeutic_game'] = {
                'engine': game_engine,
                'ui_controller': ui_controller
            }
            
            self.logger.info("Therapeutic game initialized")
            
        except Exception as e:
            self.logger.error(f"Therapeutic game initialization failed: {e}")
            raise
    
    def download_missing_models(self):
        """Download missing neural network models"""
        self.logger.info("Checking model files...")
        
        if not self.health_checker.check_model_files():
            if self.config.auto_download_models:
                self.logger.info("Downloading missing models...")
                try:
                    # Run deployment script to download models
                    import subprocess
                    result = subprocess.run([
                        "./deploy_system.sh"
                    ], capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
                    
                    if result.returncode == 0:
                        self.logger.info("Models downloaded successfully")
                    else:
                        self.logger.error(f"Model download failed: {result.stderr}")
                        raise RuntimeError("Failed to download models")
                        
                except Exception as e:
                    self.logger.error(f"Model download error: {e}")
                    raise
            else:
                raise RuntimeError("Missing models and auto-download disabled")
    
    def start_system(self):
        """Start the complete system"""
        self.logger.info("Starting Bubble system...")
        
        try:
            # Health checks
            if not self.health_checker.run_full_check():
                if not self.config.auto_download_models:
                    raise RuntimeError("System health checks failed")
            
            # Download models if needed
            self.download_missing_models()
            
            # Initialize components in order
            self.initialize_audio_system()
            self.initialize_neuro_backend()
            self.initialize_ui_system()
            
            # Start all components
            self.components['audio']['bridge'].start()
            self.components['neuro']['kernel'].start()
            
            if 'thread' in self.components.get('ui', {}):
                self.components['ui']['thread'].start()
            
            self.running = True
            self.logger.info("System started successfully!")
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            self.stop_system()
            raise
    
    def stop_system(self):
        """Gracefully stop all components"""
        self.logger.info("Stopping system...")
        
        try:
            # Stop audio
            if 'audio' in self.components:
                self.components['audio']['bridge'].stop()
            
            # UI threads are daemon, will exit automatically
            
            self.running = False
            self.logger.info("System stopped")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def run_interactive(self):
        """Run system with interactive monitoring"""
        self.start_system()
        
        try:
            print("\n=== Bubble System Running ===")
            print("Press Ctrl+C to stop")
            print(f"Mode: {self.config.mode}")
            print("Components: Audio, Neuro Backend, UI")
            
            # Simple monitoring loop
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_system()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Bubble System Unified Launcher")
    parser.add_argument("--mode", choices=["clinician", "child", "universe", "game", "enrollment", "pipeline"], 
                       required=True, help="System mode")
    parser.add_argument("--profile-name", help="Voice profile name")
    parser.add_argument("--voice-profile", help="Path to voice profile WAV file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--no-visualization", action="store_true", 
                       help="Disable visualization")
    parser.add_argument("--no-clinical-logging", action="store_true",
                       help="Disable clinical logging")
    parser.add_argument("--no-auto-download", action="store_true",
                       help="Disable automatic model downloading")
    parser.add_argument("--no-nebula", action="store_true",
                       help="Disable Cognitive Nebula 3D universe")
    parser.add_argument("--no-ai-generation", action="store_true",
                       help="Disable AI image generation")
    parser.add_argument("--universe-theme", choices=["therapeutic", "creative", "educational", "adventure"],
                       default="therapeutic", help="Universe theme")
    parser.add_argument("--ai-model", choices=["stable_diffusion", "automatic1111", "pollinations"],
                       default="stable_diffusion", help="AI image generation model")
    
    args = parser.parse_args()
    
    # Create system configuration
    config = SystemConfig(
        mode=args.mode,
        profile_name=args.profile_name,
        voice_profile_path=args.voice_profile,
        log_level=args.log_level,
        enable_visualization=not args.no_visualization,
        enable_clinical_logging=not args.no_clinical_logging,
        auto_download_models=not args.no_auto_download,
        enable_cognitive_nebula=not args.no_nebula,
        enable_ai_image_generation=not args.no_ai_generation,
        universe_theme=args.universe_theme,
        ai_model=args.ai_model
    )
    
    # Create and run orchestrator
    orchestrator = SystemOrchestrator(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal...")
        orchestrator.stop_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        orchestrator.run_interactive()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
