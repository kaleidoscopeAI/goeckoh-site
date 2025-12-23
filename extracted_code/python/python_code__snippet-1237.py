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

