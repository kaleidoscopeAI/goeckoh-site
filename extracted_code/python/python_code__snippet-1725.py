class AudioDriver:
    def __init__(self, rust_engine=None):
        self.queue = queue.Queue(maxsize=10)
        self.rust_engine = rust_engine
        self.is_running = True
        
        # Determine backend
        self.is_android = 'android' in sys.platform
        
        # Start the non-blocking consumer thread
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def play_tone_for_arousal(self, text_length: float, arousal: float):
        """
        Public API: Called by the GUI thread. 
        Offloads computation to Rust -> queues raw PCM for playback.
        """
        if self.rust_engine:
            # 1. Calls the C-Speed Rust Kernel
            # Returns a raw List[float]
            pcm_data_list = self.rust_engine.synthesize_tone_safe(text_length, arousal)
            
            # 2. Convert to efficient NumPy array (zero-copy if optimized)
            pcm_array = np.array(pcm_data_list, dtype=np.float32)
            
            # 3. Queue (Non-blocking drop if busy to prevent lag)
            if not self.queue.full():
                self.queue.put(pcm_array)

    def _playback_loop(self):
        """
        The continuous audio consumer. 
        """
        # REFLECTION: We keep the audio stream open (or trigger it efficiently).
        # Opening/closing streams per tone causes "popping" artifacts.
        
        if DESKTOP_AUDIO:
            with sd.OutputStream(channels=1, samplerate=SAMPLE_RATE) as stream:
                while self.is_running:
                    try:
                        # Blocking get wait prevents CPU spin
                        data = self.queue.get(timeout=1)
                        # Write to hardware
                        sd.play(data, SAMPLE_RATE, blocking=True) 
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Audio Error: {e}")
                        
        elif self.is_android:
            # Placeholder for Android 'plyer' or 'audiostream' implementation
            # Bulletproof: Don't crash if Android drivers are fussy.
            while self.is_running:
                try:
                    data = self.queue.get(timeout=1)
                    # Implementation detail: Android audio writes happen here
                except queue.Empty:
                    pass

