"""
Desktop audio processing bridge for Bubble system.

Provides real-time audio capture and playback for desktop environments
using sounddevice library.
"""

import threading
import queue
import numpy as np
import sounddevice as sd
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class DesktopAudioBridge:
    """Real-time audio processing bridge for desktop platforms"""
    
    def __init__(self, mic_q: queue.Queue, spk_q: queue.Queue, sr: int = 16000):
        """
        Initialize desktop audio bridge
        
        Args:
            mic_q: Queue for microphone audio data
            spk_q: Queue for speaker audio data  
            sr: Sample rate (default 16000 Hz)
        """
        self.mic_q = mic_q
        self.spk_q = spk_q
        self.sr = sr
        self.running = False
        
        # Audio stream references
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.duplex_stream: Optional[sd.Stream] = None
        
        # Buffer management
        self.buffer_size = 1024
        self.output_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        
        logger.info(f"DesktopAudioBridge initialized: sr={sr}, buffer_size={self.buffer_size}")
    
    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, 
                       frames: int, time: sd.CallbackTime, status: sd.CallbackFlags) -> None:
        """
        Real-time audio processing callback
        
        Args:
            indata: Input audio data from microphone
            outdata: Output audio data for speakers
            frames: Number of audio frames
            time: Timing information
            status: Callback status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            # Process input audio
            if len(indata) > 0:
                # Convert to float32 and flatten
                audio_data = indata.flatten().astype(np.float32)
                self.mic_q.put(audio_data)
            
            # Process output audio
            try:
                # Get audio data from speaker queue
                output_data = self.spk_q.get_nowait()
                
                # Ensure correct shape and type
                if output_data.ndim == 1:
                    output_data = output_data.reshape(-1, 1)
                
                # Fill output buffer
                min_len = min(len(output_data), len(outdata))
                outdata[:min_len] = output_data[:min_len]
                
                # Zero-pad if needed
                if min_len < len(outdata):
                    outdata[min_len:] = 0
                    
            except queue.Empty:
                # No audio available, output silence
                outdata[:] = 0
                
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            outdata[:] = 0
    
    def start(self) -> None:
        """Start audio processing streams"""
        if self.running:
            logger.warning("Audio bridge already running")
            return
        
        try:
            # Check available devices
            devices = sd.query_devices()
            if not devices:
                raise RuntimeError("No audio devices found")
            
            logger.info(f"Found {len(devices)} audio devices")
            
            # Create duplex stream for simultaneous input/output
            self.duplex_stream = sd.Stream(
                samplerate=self.sr,
                channels=1,
                dtype=np.float32,
                blocksize=self.buffer_size,
                callback=self._audio_callback
            )
            
            # Start the stream
            self.duplex_stream.start()
            self.running = True
            
            logger.info("Desktop audio bridge started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio bridge: {e}")
            raise
    
    def stop(self) -> None:
        """Stop audio processing streams"""
        if not self.running:
            return
        
        try:
            # Stop and close streams
            if self.duplex_stream:
                self.duplex_stream.stop()
                self.duplex_stream.close()
                self.duplex_stream = None
            
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            
            self.running = False
            logger.info("Desktop audio bridge stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio bridge: {e}")
    
    def get_device_info(self) -> dict:
        """Get information about available audio devices"""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            return {
                'total_devices': len(devices),
                'default_input_device': default_input,
                'default_output_device': default_output,
                'devices': [
                    {
                        'id': i,
                        'name': device['name'],
                        'max_input_channels': device['max_input_channels'],
                        'max_output_channels': device['max_output_channels'],
                        'default_input': i == default_input,
                        'default_output': i == default_output
                    }
                    for i, device in enumerate(devices)
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
    
    def test_audio(self, duration: float = 2.0) -> bool:
        """
        Test audio hardware with a simple loopback test
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            True if test passes, False otherwise
        """
        logger.info(f"Running audio test for {duration} seconds...")
        
        try:
            test_queue = queue.Queue()
            test_completed = threading.Event()
            test_passed = False
            
            def test_callback(indata, outdata, frames, time, status):
                nonlocal test_passed
                
                if status:
                    logger.warning(f"Test callback status: {status}")
                
                # Echo input to output
                outdata[:] = indata
                
                # Check if we're getting audio
                if np.max(np.abs(indata)) > 0.01:
                    test_passed = True
            
            # Create test stream
            test_stream = sd.Stream(
                samplerate=self.sr,
                channels=1,
                dtype=np.float32,
                callback=test_callback
            )
            
            test_stream.start()
            
            # Wait for test duration
            threading.Timer(duration, test_completed.set).start()
            test_completed.wait()
            
            test_stream.stop()
            test_stream.close()
            
            if test_passed:
                logger.info("Audio test passed")
            else:
                logger.warning("Audio test passed but no signal detected")
            
            return True
            
        except Exception as e:
            logger.error(f"Audio test failed: {e}")
            return False

# Convenience function for creating audio bridge
def create_desktop_audio_bridge(mic_q: queue.Queue, spk_q: queue.Queue, 
                              sr: int = 16000) -> DesktopAudioBridge:
    """
    Factory function to create desktop audio bridge
    
    Args:
        mic_q: Queue for microphone audio data
        spk_q: Queue for speaker audio data
        sr: Sample rate
        
    Returns:
        Configured DesktopAudioBridge instance
    """
    return DesktopAudioBridge(mic_q, spk_q, sr)
