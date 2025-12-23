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
