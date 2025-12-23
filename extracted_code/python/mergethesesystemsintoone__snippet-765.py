def __init__(self, task_manager: TaskManager, update_interval: float = 1.0):
    self.task_manager = task_manager
    self.update_interval = update_interval
    self.running = False
    self.monitor_task = None
    self.start_time = time.time()
    self.time_points = []
    self.cpu_history = []
    self.memory_history = []

async def start(self) -> None:
    if not self.running:
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitor started")

async def stop(self) -> None:
    if self.running:
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            await self.monitor_task
        logger.info("System monitor stopped")

async def _monitor_loop(self) -> None:
    try:
        while self.running:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            elapsed = time.time() - self.start_time
            self.time_points.append(elapsed)
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_percent)
            task_status = {
                'Pending': len([t for t in self.task_manager.tasks.values() if t.status == 'pending']),
                'Running': len(self.task_manager.running_tasks),
                'Completed': len(self.task_manager.completed_tasks),
                'Failed': len(self.task_manager.failed_tasks)
            }
            print(f"\r{Fore.CYAN}Tasks: {Fore.GREEN}{task_status['Completed']} completed{Fore.RESET}, "
                  f"{Fore.YELLOW}{task_status['Running']} running{Fore.RESET}, "
                  f"{Fore.BLUE}{task_status['Pending']} pending{Fore.RESET}, "
                  f"{Fore.RED}{task_status['Failed']} failed{Fore.RESET} | "
                  f"CPU: {cpu_percent}%, Mem: {memory_percent}%", end='')
            await asyncio.sleep(self.update_interval)
    except asyncio.CancelledError:
        logger.info("Monitor loop cancelled")

