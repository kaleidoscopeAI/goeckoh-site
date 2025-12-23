"""
Production-ready system monitoring and health checks

Provides comprehensive monitoring of the Bubble system including
performance metrics, component health, and alerting.
"""

import time
import threading
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import queue

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    audio_latency_ms: float
    processing_fps: float
    queue_sizes: Dict[str, int]
    component_status: Dict[str, bool]
    error_count: int
    warning_count: int

@dataclass 
class Alert:
    """System alert notification"""
    timestamp: datetime
    severity: str  # "critical", "warning", "info"
    component: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

class SystemMonitor:
    """Comprehensive system monitoring"""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.running = False
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[Alert] = []
        self.max_history = 3600  # 1 hour at 1 second intervals
        
        # Component health callbacks
        self.component_checks: Dict[str, Callable] = {}
        
        # Thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'audio_latency_ms': 100.0,
            'processing_fps': 15.0,
            'queue_size': 1000,
            'error_rate': 10.0  # errors per minute
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.last_check_time = time.time()
        self.frame_count = 0
        self.error_count = 0
        self.warning_count = 0
    
    def register_component_check(self, component: str, check_func: Callable):
        """Register a component health check function"""
        self.component_checks[component] = check_func
        self.logger.info(f"Registered health check for component: {component}")
    
    def register_alert_callback(self, callback: Callable):
        """Register an alert notification callback"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Update frame rate
                self.frame_count += 1
                
                # Sleep until next check
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self.error_count += 1
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # System resource metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Component status
        component_status = {}
        for component, check_func in self.component_checks.items():
            try:
                component_status[component] = bool(check_func())
            except Exception:
                component_status[component] = False
        
        # Audio queue sizes (if available)
        queue_sizes = {}
        try:
            # Try to get queue sizes from common components
            import sys
            if 'system_launcher' in sys.modules:
                # Access system components if available
                pass  # Implementation would depend on system structure
        except Exception:
            pass
        
        # Calculate processing FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_check_time) if current_time > self.last_check_time else 0.0
        self.last_check_time = current_time
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            audio_latency_ms=0.0,  # Would be measured from audio pipeline
            processing_fps=fps,
            queue_sizes=queue_sizes,
            component_status=component_status,
            error_count=self.error_count,
            warning_count=self.warning_count
        )
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(Alert(
                timestamp=metrics.timestamp,
                severity="warning",
                component="system",
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                metric_value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_percent']
            ))
        
        # Memory usage alert
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(Alert(
                timestamp=metrics.timestamp,
                severity="warning",
                component="system",
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                metric_value=metrics.memory_percent,
                threshold=self.thresholds['memory_percent']
            ))
        
        # Component failure alerts
        for component, status in metrics.component_status.items():
            if not status:
                alerts.append(Alert(
                    timestamp=metrics.timestamp,
                    severity="critical",
                    component=component,
                    message=f"Component failure: {component}"
                ))
        
        # Processing FPS alert
        if metrics.processing_fps < self.thresholds['processing_fps']:
            alerts.append(Alert(
                timestamp=metrics.timestamp,
                severity="warning",
                component="system",
                message=f"Low processing FPS: {metrics.processing_fps:.1f}",
                metric_value=metrics.processing_fps,
                threshold=self.thresholds['processing_fps']
            ))
        
        # Queue size alerts
        for queue_name, size in metrics.queue_sizes.items():
            if size > self.thresholds['queue_size']:
                alerts.append(Alert(
                    timestamp=metrics.timestamp,
                    severity="warning",
                    component="audio",
                    message=f"Large queue size: {queue_name} = {size}",
                    metric_value=size,
                    threshold=self.thresholds['queue_size']
                ))
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """Send alert notification"""
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        # Log alert
        log_level = {
            "critical": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(log_level, f"[{alert.severity.upper()}] {alert.component}: {alert.message}")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_recent_alerts(self, minutes: int = 10) -> List[Alert]:
        """Get alerts from recent minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def export_metrics(self, filepath: str, minutes: int = 60) -> bool:
        """Export metrics to JSON file"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_metrics = [
                asdict(metric) for metric in self.metrics_history
                if metric.timestamp >= cutoff_time
            ]
            
            # Convert datetime objects to strings
            for metric in recent_metrics:
                metric['timestamp'] = metric['timestamp'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(recent_metrics, f, indent=2)
            
            self.logger.info(f"Exported {len(recent_metrics)} metrics to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False

# Global monitor instance
_global_monitor: Optional[SystemMonitor] = None

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor

def start_system_monitoring():
    """Start global system monitoring"""
    monitor = get_system_monitor()
    monitor.start_monitoring()

def stop_system_monitoring():
    """Stop global system monitoring"""
    monitor = get_system_monitor()
    monitor.stop_monitoring()
