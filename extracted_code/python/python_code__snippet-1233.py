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

