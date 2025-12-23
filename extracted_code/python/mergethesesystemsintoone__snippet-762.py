# Enhanced with resource prediction
def __init__(self, task_manager: TaskManager, update_interval: float = 1.0):
    self.task_manager = task_manager
    self.update_interval = update_interval
    self.running = False
    self.monitor_task = None
    self.start_time = time.time()

    self.interactive_mode = hasattr(plt, 'ion')
    if self.interactive_mode:
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)

        self.time_points = []
        self.cpu_history = []
        self.memory_history = []
        self.task_counts = []

        self.cpu_line, = self.ax1.plot([], [], 'r-', label='CPU Usage (%)')
        self.memory_line, = self.ax1.plot([], [], 'b-', label='Memory Usage (%)')
        self.task_bars = self.ax2.bar([], [])

        self.ax1.set_title('System Resource Usage')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Usage (%)')
        self.ax1.set_ylim(0, 100)
        self.ax1.grid(True)
        self.ax1.legend()

        self.ax2.set_title('Task Status')
        self.ax2.set_ylabel('Number of Tasks')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # New: Resource prediction
    self.resource_predictions = []

async def _predict_resources(self) -> Dict[str, float]:
    """Simple resource usage prediction based on historical data"""
    if len(self.cpu_history) < 5:
        return {'cpu': 0.0, 'memory': 0.0}

    cpu_trend = np.polyfit(self.time_points[-5:], self.cpu_history[-5:], 1)[0]
    mem_trend = np.polyfit(self.time_points[-5:], self.memory_history[-5:], 1)[0]

    return {
        'cpu': max(0, min(100, self.cpu_history[-1] + cpu_trend * self.update_interval)),
        'memory': max(0, min(100, self.memory_history[-1] + mem_trend * self.update_interval))
    }

async def _monitor_loop(self) -> None:
    try:
        while self.running:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            logger.debug(f"Resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%")

            elapsed = time.time() - self.start_time
            if self.interactive_mode:
                self.time_points.append(elapsed)
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)

                task_status = {
                    'Pending': len([t for t in self.task_manager.tasks.values() if t.status == 'pending']),
                    'Running': len(self.task_manager.running_tasks),
                    'Completed': len(self.task_manager.completed_tasks),
                    'Failed': len(self.task_manager.failed_tasks)
                }
                self._update_plots(task_status)

            pending_count = len([t for t in self.task_manager.tasks.values() if t.status == 'pending'])
            running_count = len(self.task_manager.running_tasks)
            completed_count = len(self.task_manager.completed_tasks)
            failed_count = len(self.task_manager.failed_tasks)

            # New: Resource prediction
            prediction = await self._predict_resources()
            print(f"\r{Fore.CYAN}Tasks: {Fore.GREEN}{completed_count} completed{Fore.RESET}, "
                  f"{Fore.YELLOW}{running_count} running{Fore.RESET}, "
                  f"{Fore.BLUE}{pending_count} pending{Fore.RESET}, "
                  f"{Fore.RED}{failed_count} failed{Fore.RESET} | "
                  f"CPU: {cpu_percent}% (Pred: {prediction['cpu']:.1f}%), "
                  f"Mem: {memory_percent}% (Pred: {prediction['memory']:.1f}%)", end='')

            await asyncio.sleep(self.update_interval)

    except asyncio.CancelledError:
        logger.info("Monitor loop cancelled")
    except Exception as e:
        logger.error(f"Error in monitor loop: {e}", exc_info=True)

# ... (rest of SystemMonitor remains similar)

