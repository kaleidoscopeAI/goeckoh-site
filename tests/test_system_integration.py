"""
Comprehensive system integration tests

Tests the complete Bubble system integration including
audio processing, neural backend, UI components, and
configuration validation.
"""

import unittest
import threading
import time
import queue
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "test_config.yaml"
        
        # Create minimal test config
        self.test_config = {
            "audio": {
                "sample_rate": 16000,
                "buffer_size": 1024
            },
            "models": {
                "tokens": "test_tokens.txt",
                "encoder": "test_encoder.onnx",
                "decoder": "test_decoder.onnx",
                "tts": "test_tts.onnx"
            },
            "ui": {
                "bubble_radius": 100.0,
                "update_rate": 30.0
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def test_audio_desktop_bridge_creation(self):
        """Test desktop audio bridge creation and basic functionality"""
        try:
            from src.audio_desktop import DesktopAudioBridge
            
            # Create test queues
            mic_q = queue.Queue()
            spk_q = queue.Queue()
            
            # Create audio bridge
            bridge = DesktopAudioBridge(mic_q, spk_q, sr=16000)
            
            # Test basic properties
            self.assertEqual(bridge.sr, 16000)
            self.assertFalse(bridge.running)
            self.assertIsNotNone(bridge.buffer_size)
            
            # Test device info
            device_info = bridge.get_device_info()
            self.assertIsInstance(device_info, dict)
            
            logger.info("✓ Desktop audio bridge creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Desktop audio bridge not available: {e}")
        except Exception as e:
            self.fail(f"Desktop audio bridge test failed: {e}")
    
    def test_ui_foam_widget_creation(self):
        """Test UI foam widget creation and properties"""
        try:
            from src.ui_foam import FoamWidget
            
            # Create widget
            widget = FoamWidget()
            
            # Test default properties
            self.assertEqual(widget.rms_energy, 0.0)
            self.assertEqual(widget.gcl_coherence, 1.0)
            self.assertEqual(widget.entropy, 0.0)
            self.assertEqual(widget.bubble_radius, 100.0)
            self.assertEqual(len(widget.bubble_color), 4)  # RGBA
            
            # Test physics update
            widget.update_physics(0.1, 0.5, 0.2)
            self.assertEqual(widget.rms_energy, 0.1)
            self.assertEqual(widget.gcl_coherence, 0.5)
            self.assertEqual(widget.entropy, 0.2)
            
            logger.info("✓ UI foam widget creation test passed")
            
        except ImportError as e:
            self.skipTest(f"UI foam widget not available: {e}")
        except Exception as e:
            self.fail(f"UI foam widget test failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation system"""
        try:
            from config_validator import ConfigValidator, ConfigIssue
            
            # Create validator
            validator = ConfigValidator(str(self.config_path), "config.schema.yaml")
            
            # Load test config
            config = validator.load_config()
            self.assertIsInstance(config, dict)
            
            # Test config saving
            config['test_field'] = 'test_value'
            success = validator.save_config(config)
            self.assertTrue(success)
            
            # Verify saved config
            reloaded_config = validator.load_config()
            self.assertEqual(reloaded_config['test_field'], 'test_value')
            
            logger.info("✓ Configuration validation test passed")
            
        except ImportError as e:
            self.skipTest(f"Configuration validator not available: {e}")
        except Exception as e:
            self.fail(f"Configuration validation test failed: {e}")
    
    def test_system_launcher_initialization(self):
        """Test system launcher initialization"""
        try:
            from system_launcher import SystemOrchestrator, SystemConfig
            
            # Create test configuration
            config = SystemConfig(
                mode="clinician",
                log_level="INFO",
                enable_visualization=False,  # Disable for testing
                auto_download_models=False
            )
            
            # Create orchestrator
            orchestrator = SystemOrchestrator(config)
            
            # Test basic properties
            self.assertEqual(orchestrator.config.mode, "clinician")
            self.assertFalse(orchestrator.running)
            self.assertIsNotNone(orchestrator.logger)
            
            # Test health checker
            health_checker = orchestrator.health_checker
            self.assertIsNotNone(health_checker)
            
            logger.info("✓ System launcher initialization test passed")
            
        except ImportError as e:
            self.skipTest(f"System launcher not available: {e}")
        except Exception as e:
            self.fail(f"System launcher initialization test failed: {e}")
    
    def test_audio_queue_functionality(self):
        """Test audio queue processing"""
        # Create test queues
        mic_q = queue.Queue()
        spk_q = queue.Queue()
        
        # Test queue operations
        test_audio = np.random.randn(1024).astype(np.float32)
        mic_q.put(test_audio)
        
        # Verify queue contains data
        self.assertFalse(mic_q.empty())
        retrieved_audio = mic_q.get()
        self.assertTrue(np.array_equal(test_audio, retrieved_audio))
        self.assertTrue(mic_q.empty())
        
        logger.info("✓ Audio queue functionality test passed")
    
    def test_physics_calculations(self):
        """Test physics calculations for bubble visualization"""
        try:
            from heart import CrystallineHeart
            
            # Create heart component
            heart = CrystallineHeart()
            
            # Test pulse calculation
            gcl, entropy = heart.pulse(0.1, 0.05)  # energy, latency
            
            # Validate ranges
            self.assertGreaterEqual(gcl, 0.0)
            self.assertLessEqual(gcl, 1.0)
            self.assertGreaterEqual(entropy, 0.0)
            
            # Test state persistence
            initial_state = heart.state.copy()
            heart.pulse(0.2, 0.1)
            self.assertFalse(np.array_equal(initial_state, heart.state))
            
            logger.info("✓ Physics calculations test passed")
            
        except ImportError as e:
            self.skipTest(f"Heart component not available: {e}")
        except Exception as e:
            self.fail(f"Physics calculations test failed: {e}")
    
    def test_grammar_correction(self):
        """Test grammar correction functionality"""
        try:
            from grammar import correct_text
            
            # Test basic correction
            result = correct_text("you are happy", 0.5)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            
            # Test safety mode (low GCL)
            result_safe = correct_text("you are happy", 0.3)
            self.assertEqual(result_safe, "you are happy")  # Should return unchanged
            
            # Test empty input
            result_empty = correct_text("", 0.5)
            self.assertIsNone(result_empty)
            
            logger.info("✓ Grammar correction test passed")
            
        except ImportError as e:
            self.skipTest(f"Grammar correction not available: {e}")
        except Exception as e:
            self.fail(f"Grammar correction test failed: {e}")
    
    def test_behavior_monitoring(self):
        """Test behavior monitoring system"""
        try:
            from behavior import BehaviorMonitor, StrategyAdvisor
            
            # Create behavior monitor
            monitor = BehaviorMonitor()
            
            # Test registration
            event = monitor.register("hello world", False, 0.05)
            # Event might be None, that's okay
            
            # Test multiple corrections
            for i in range(4):
                event = monitor.register("test", True, 0.08)
            
            # Should trigger anxious event
            self.assertEqual(event, "anxious")
            
            # Test strategy advisor
            advisor = StrategyAdvisor()
            strategies = advisor.suggest("anxious")
            self.assertIsNotNone(strategies)
            self.assertGreater(len(strategies), 0)
            
            logger.info("✓ Behavior monitoring test passed")
            
        except ImportError as e:
            self.skipTest(f"Behavior monitoring not available: {e}")
        except Exception as e:
            self.fail(f"Behavior monitoring test failed: {e}")
    
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

class TestSystemPerformance(unittest.TestCase):
    """Test system performance under load"""
    
    def test_audio_processing_performance(self):
        """Test audio processing performance"""
        try:
            from src.audio_desktop import DesktopAudioBridge
            
            mic_q = queue.Queue()
            spk_q = queue.Queue()
            bridge = DesktopAudioBridge(mic_q, spk_q)
            
            # Generate test audio data
            test_audio = np.random.randn(16000).astype(np.float32)  # 1 second
            
            # Time queue operations
            start_time = time.time()
            for _ in range(100):
                mic_q.put(test_audio[:1024])
                if not spk_q.empty():
                    spk_q.get()
            
            processing_time = time.time() - start_time
            
            # Should process 100 chunks in reasonable time
            self.assertLess(processing_time, 1.0)  # Less than 1 second
            
            logger.info(f"✓ Audio processing performance: {processing_time:.3f}s for 100 chunks")
            
        except ImportError:
            self.skipTest("Desktop audio bridge not available")
    
    def test_physics_calculation_performance(self):
        """Test physics calculation performance"""
        try:
            from heart import CrystallineHeart
            
            heart = CrystallineHeart()
            
            # Time physics calculations
            start_time = time.time()
            for _ in range(1000):
                heart.pulse(0.1, 0.05)
            
            calculation_time = time.time() - start_time
            
            # Should complete 1000 calculations quickly
            self.assertLess(calculation_time, 0.1)  # Less than 100ms
            
            logger.info(f"✓ Physics calculation performance: {calculation_time:.3f}s for 1000 calculations")
            
        except ImportError:
            self.skipTest("Heart component not available")

def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = run_integration_tests()
    sys.exit(0 if success else 1)
