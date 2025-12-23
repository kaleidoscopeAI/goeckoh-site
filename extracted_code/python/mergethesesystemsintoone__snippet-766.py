    self.max_memory_percent = max_memory_percent
    self.resource_lock = threading.Lock()

    # Keep track of allocated resources
    self.allocated_cpu = 0.0  # percentage points
    self.allocated_memory = 0.0  # percentage points

    # Monitor thread
    self.stop_event = threading.Event()
    self.monitor_thread = threading.Thread(target=self._resource_monitor_loop, daemon=True)
    self.monitor_thread.start()

def get_available_resources(self) -> Dict[str, float]:
    """Get available system resources"""
    with self.resource_lock:
        # Get current system resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # Calculate available resources
        available_cpu = max(0.0, self.max_cpu_percent - cpu_percent - self.allocated_cpu)
        available_memory = max(0.0, self.max_memory_percent - memory_percent - self.allocated_memory)

        return {
            "cpu_percent": available_cpu,
            "memory_percent": available_memory,
            "system_cpu_percent": cpu_percent,
            "system_memory_percent": memory_percent
        }

def allocate_resources(self, resources: Dict[str, float]) -> bool:
    """
    Try to allocate resources for a task

    Args:
        resources: Resource requirements (cpu_percent, memory_percent)

    Returns:
        Whether resources were successfully allocated
    """
    with self.resource_lock:
        # Check available resources
        available = self.get_available_resources()

        cpu_required = resources.get("cpu_percent", 0.0)
        memory_required = resources.get("memory_percent", 0.0)

        # Check if we have enough resources
        if (cpu_required > available["cpu_percent"] or 
            memory_required > available["memory_percent"]):
            return False

        # Allocate resources
        self.allocated_cpu += cpu_required
        self.allocated_memory += memory_required

        return True

def release_resources(self, resources: Dict[str, float]):
    """Release allocated resources"""
    with self.resource_lock:
        cpu_allocated = resources.get("cpu_percent", 0.0)
        memory_allocated = resources.get("memory_percent", 0.0)

        self.allocated_cpu = max(0.0, self.allocated_cpu - cpu_allocated)
        self.allocated_memory = max(0.0, self.allocated_memory - memory_allocated)

def _resource_monitor_loop(self):
    """Background thread to monitor system resources"""
    while not self.stop_event.is_set():
        try:
            # Get current system resource usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_percent = psutil.virtual_memory().percent

            # Log if resources are getting low
            if cpu_percent > self.max_cpu_percent - 10:
                logger.warning(f"System CPU usage is high: {cpu_percent}%")

            if memory_percent > self.max_memory_percent - 10:
                logger.warning(f"System memory usage is high: {memory_percent}%")

            # Sleep before next check
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in resource monitor: {str(e)}")
            time.sleep(10)  # Sleep longer on error

def stop(self):
    """Stop the resource monitor"""
    self.stop_event.set()
    if self.monitor_thread.is_alive():
        self.monitor_thread.join(timeout=2.0)

