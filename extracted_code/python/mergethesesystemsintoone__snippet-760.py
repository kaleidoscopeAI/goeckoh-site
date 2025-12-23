def __init__(self, config: TaskManagerConfig):
    # Existing initialization
    self.config = config
    self.tasks: Dict[str, TaskConfig] = {}
    self.task_queue: List[str] = []
    self.running_tasks: Set[str] = set()
    self.completed_tasks: Dict[str, TaskConfig] = {}
    self.failed_tasks: Dict[str, TaskConfig] = {}

    self.work_dir = Path(config.work_dir)
    self.work_dir.mkdir(exist_ok=True, parents=True)
    self.tasks_dir = self.work_dir / "tasks"
    self.tasks_dir.mkdir(exist_ok=True)
    self.results_dir = self.work_dir / "results"
    self.results_dir.mkdir(exist_ok=True)

    self.loop = asyncio.get_event_loop()

    self.system_resources = {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_total': psutil.disk_usage('/').total,
        'disk_free': psutil.disk_usage('/').free,
    }

    self.process_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=config.max_concurrent_tasks
    )

    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)

    # New: Backup directory and state
    self.backup_dir = self.work_dir / "backups"
    self.backup_dir.mkdir(exist_ok=True)
    self._load_state()  # Load previous state if available

    logger.info(f"Task Manager initialized with config: {config}")
    logger.info(f"System resources: {self.system_resources}")

def _load_state(self) -> None:
    """Load previous state from backup"""
    latest_backup = max(
        (f for f in self.backup_dir.glob("state_*.json")),
        key=lambda x: x.stat().st_mtime,
        default=None
    )
    if latest_backup:
        try:
            with open(latest_backup, 'r') as f:
                state = json.load(f)
            self.tasks.update({t['task_id']: TaskConfig(**t) for t in state.get('tasks', [])})
            self.completed_tasks.update({t['task_id']: TaskConfig(**t) for t in state.get('completed_tasks', [])})
            self.failed_tasks.update({t['task_id']: TaskConfig(**t) for t in state.get('failed_tasks', [])})
            logger.info(f"Loaded state from {latest_backup}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

async def _backup_state(self) -> None:
    """Periodically backup the current state"""
    while True:
        try:
            state = {
                'tasks': [t.to_dict() for t in self.tasks.values()],
                'completed_tasks': [t.to_dict() for t in self.completed_tasks.values()],
                'failed_tasks': [t.to_dict() for t in self.failed_tasks.values()],
                'timestamp': datetime.now().isoformat()
            }
            backup_file = self.backup_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"State backed up to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to backup state: {e}")
        await asyncio.sleep(self.config.backup_interval)

# Existing methods with enhancements
def add_task(self, task: TaskConfig) -> str:
    for dep_id in task.dependencies:
        if dep_id not in self.tasks and dep_id not in self.completed_tasks:
            raise ValueError(f"Dependency {dep_id} does not exist")

    self.tasks[task.task_id] = task
    self._save_task_config(task)
    self._update_task_queue()

    # New: Notify via webhook
    if self.config.webhook_url:
        asyncio.create_task(self._send_webhook({
            'event': 'task_added',
            'task_id': task.task_id,
            'task_name': task.task_name,
            'timestamp': datetime.now().isoformat()
        }))

    logger.info(f"Added task: {task.task_name} (ID: {task.task_id})")
    return task.task_id

async def _send_webhook(self, payload: Dict[str, Any]) -> None:
    """Send notification to webhook URL"""
    if not self.config.webhook_url:
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.config.webhook_url, json=payload) as resp:
                if resp.status != 200:
                    logger.warning(f"Webhook notification failed: {resp.status}")
    except Exception as e:
        logger.error(f"Webhook notification error: {e}")

async def run(self) -> bool:
    logger.info("Starting task execution")

    # Start backup task
    backup_task = asyncio.create_task(self._backup_state())

    try:
        start_time = time.time()
        total_tasks = len(self.tasks)
        completed = 0
        failed = 0

        if self.config.show_progress:
            progress = tqdm(total=total_tasks, desc="Processing tasks")

        while self.tasks or self.running_tasks:
            self._update_task_queue()
            resources_available = self._check_resources()

            while (self.task_queue and 
                   len(self.running_tasks) < self.config.max_concurrent_tasks and 
                   resources_available):

                task_id = self.task_queue.pop(0)
                task = self.tasks[task_id]
                asyncio.create_task(self._execute_task(task))
                self.running_tasks.add(task_id)
                resources_available = self._check_resources()

            if self.config.show_progress:
                current_completed = len(self.completed_tasks)
                current_failed = len(self.failed_tasks)
                if current_completed + current_failed > completed + failed:
                    progress.update(current_completed + current_failed - completed - failed)
                    completed = current_completed
                    failed = current_failed

            await asyncio.sleep(0.1)

        if self.config.show_progress:
            progress.close()

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Task execution completed in {duration:.2f} seconds")
        logger.info(f"Tasks completed: {len(self.completed_tasks)}")
        logger.info(f"Tasks failed: {len(self.failed_tasks)}")

        self._generate_report()
        return len(self.failed_tasks) == 0

    except Exception as e:
        logger.error(f"Error during task execution: {e}", exc_info=True)
        return False
    finally:
        backup_task.cancel()

async def _execute_task(self, task: TaskConfig) -> None:
    logger.info(f"Starting task: {task.task_name} (ID: {task.task_id})")

    task.status = "running"
    task.start_time = time.time()
    self._save_task_config(task)

    try:
        env = os.environ.copy()
        env.update(task.environment)

        working_dir = task.working_dir or str(self.work_dir)
        output_file = None
        if task.output_file:
            output_path = Path(working_dir) / task.output_file
            output_file = open(output_path, 'w')

        if task.command:
            timeout = task.timeout or None
            process = await asyncio.create_subprocess_exec(
                *task.command,
                stdout=output_file if output_file else asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=working_dir
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Task {task.task_name} exceeded timeout of {timeout} seconds")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Task failed: {task.task_name} - {error_msg}")
                task.status = "failed"
                task.error_message = error_msg

                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = "pending"
                    logger.info(f"Retrying task {task.task_name} ({task.retry_count}/{task.max_retries})")
                    await asyncio.sleep(self.config.retry_delay)  # Add delay before retry
                else:
                    self.failed_tasks[task.task_id] = task
                    del self.tasks[task.task_id]
            else:
                task.status = "completed"
                task.result = stdout.decode() if stdout else None
                self.completed_tasks[task.task_id] = task
                del self.tasks[task.task_id]
        else:
            task.status = "completed"
            self.completed_tasks[task.task_id] = task
            del self.tasks[task.task_id]

        if output_file:
            output_file.close()

    except Exception as e:
        logger.error(f"Error executing task {task.task_name}: {e}", exc_info=True)
        task.status = "failed"
        task.error_message = str(e)

        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = "pending"
            logger.info(f"Retrying task {task.task_name} ({task.retry_count}/{task.max_retries})")
            await asyncio.sleep(self.config.retry_delay)
        else:
            self.failed_tasks[task.task_id] = task
            del self.tasks[task.task_id]

    finally:
        task.end_time = time.time()
        self._save_task_config(task)
        self.running_tasks.remove(task.task_id)

        if task.status == "failed" and self.config.abort_on_fail:
            logger.critical(f"Aborting due to task failure: {task.task_name}")
            self._abort_all_tasks()

        # New: Notify via webhook
        if self.config.webhook_url:
            await self._send_webhook({
                'event': 'task_completed',
                'task_id': task.task_id,
                'task_name': task.task_name,
                'status': task.status,
                'timestamp': datetime.now().isoformat()
            })

# Remaining methods (TaskManager) remain largely unchanged, just adding webhook notifications where appropriate
def _save_task_config(self, task: TaskConfig) -> None:
    task_file = self.tasks_dir / f"{task.task_id}.json"
    with open(task_file, 'w') as f:
        json.dump(task.to_dict(), f, indent=2)

def _update_task_queue(self) -> None:
    self.task_queue = []
    for task_id, task in self.tasks.items():
        if task_id not in self.running_tasks and task.status == "pending":
            deps_satisfied = all(
                dep_id in self.completed_tasks or 
                (dep_id in self.tasks and self.tasks[dep_id].status == "completed")
                for dep_id in task.dependencies
            )
            if deps_satisfied:
                self.task_queue.append(task_id)
    self.task_queue.sort(key=lambda task_id: self.tasks[task_id].priority, reverse=True)
    logger.debug(f"Updated task queue: {self.task_queue}")

def _check_resources(self) -> bool:
    memory = psutil.virtual_memory()
    memory_used_fraction = memory.percent / 100.0
    cpu_used_fraction = psutil.cpu_percent(interval=None) / 100.0
    memory_available = memory_used_fraction < self.config.max_memory_usage
    cpu_available = cpu_used_fraction < self.config.max_cpu_usage
    logger.debug(f"Resource check - Memory: {memory_used_fraction:.2f}/{self.config.max_memory_usage}, "
                f"CPU: {cpu_used_fraction:.2f}/{self.config.max_cpu_usage}")
    return memory_available and cpu_available

def _abort_all_tasks(self) -> None:
    logger.warning("Aborting all tasks")
    for task_id in list(self.running_tasks):
        task = self.tasks[task_id]
        task.status = "failed"
        task.error_message = "Aborted due to failure in dependent task"
        self.failed_tasks[task_id] = task
        del self.tasks[task_id]
    self.running_tasks.clear()
    for task_id, task in list(self.tasks.items()):
        if task.status == "pending":
            task.status = "failed"
            task.error_message = "Aborted due to failure in dependent task"
            self.failed_tasks[task_id] = task
            del self.tasks[task_id]

def _generate_report(self) -> Dict[str, Any]:
    report = {
        "timestamp": datetime.now().isoformat(),
        "duration": sum(
            (task.end_time or 0) - (task.start_time or 0)
            for task in list(self.completed_tasks.values()) + list(self.failed_tasks.values())
        ),
        "total_tasks": len(self.completed_tasks) + len(self.failed_tasks),
        "completed_tasks": len(self.completed_tasks),
        "failed_tasks": len(self.failed_tasks),
        "success_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) if len(self.completed_tasks) + len(self.failed_tasks) > 0 else 0,
        "tasks": {
            "completed": [task.to_dict() for task in self.completed_tasks.values()],
            "failed": [task.to_dict() for task in self.failed_tasks.values()]
        }
    }
    report_file = self.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Task report generated and saved to {report_file}")
    return report

def _signal_handler(self, sig, frame) -> None:
    logger.warning(f"Received signal {sig}, shutting down gracefully...")
    self._generate_report()
    sys.exit(0)

