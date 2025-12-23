def __init__(self, max_workers: Optional[int] = None, persist_path: Optional[str] = None, auto_recovery: bool = True):
    self.max_workers = max_workers or MAX_WORKERS
    self.persist_path = persist_path or TASK_PERSIST_PATH
    self.auto_recovery = auto_recovery
    self.tasks: Dict[str, Task] = {}
    self.task_queues = {
        TaskPriority.LOW: queue.PriorityQueue(),
        TaskPriority.NORMAL: queue.PriorityQueue(),
        TaskPriority.HIGH: queue.PriorityQueue(),
        TaskPriority.CRITICAL: queue.PriorityQueue()
    }
    self.running_tasks: Dict[str, asyncio.Task] = {}
    self.dependency_map: Dict[str, List[str]] = {}
    self.task_lock = threading.Lock()
    self.stop_event = threading.Event()
    self.resource_monitor = ResourceMonitor()
    self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
    self.loop = asyncio.new_event_loop()
    self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
    self.scheduler_thread.start()
    if self.persist_path and os.path.exists(self.persist_path):
        self._load_tasks()
    logger.info(f"Task scheduler initialized with {self.max_workers} workers")

def add_task(self, name: str, func: Callable, args: List = None, kwargs: Dict[str, Any] = None,
             priority: TaskPriority = TaskPriority.NORMAL, timeout_seconds: int = 3600,
             dependencies: List[str] = None, owner: Optional[str] = None, metadata: Dict[str, Any] = None,
             estimated_resources: Dict[str, float] = None) -> str:
    task_id = str(uuid.uuid4())
    estimated_resources = estimated_resources or {"cpu_percent": 25.0, "memory_percent": 10.0}
    task = Task(
        task_id=task_id, name=name, func=func, args=args or [], kwargs=kwargs or {},
        priority=priority, timeout_seconds=timeout_seconds, dependencies=dependencies or [],
        owner=owner, metadata=metadata or {}, estimated_resources=estimated_resources
    )
    with self.task_lock:
        self.tasks[task_id] = task
        for dep_id in task.dependencies:
            self.dependency_map.setdefault(dep_id, []).append(task_id)
        if not task.dependencies:
            self._enqueue_task(task)
        if self.persist_path:
            self._save_tasks()
    logger.info(f"Added task {task_id} ({name}) with priority {priority.name}")
    return task_id

def _enqueue_task(self, task: Task):
    queue_item = (-task.priority.value, task.created_at.timestamp(), task.task_id)
    self.task_queues[task.priority].put(queue_item)

def _scheduler_loop(self):
    asyncio.set_event_loop(self.loop)
    while not self.stop_event.is_set():
        try:
            with self.task_lock:
                if len(self.running_tasks) >= self.max_workers:
                    time.sleep(0.1)
                    continue
            task_id = None
            for priority in reversed(sorted(self.task_queues.keys(), key=lambda p: p.value)):
                queue = self.task_queues[priority]
                if not queue.empty():
                    try:
                        _, _, task_id = queue.get_nowait()
                        break
                    except queue.Empty:
                        pass
            if not task_id:
                time.sleep(0.1)
                continue
            with self.task_lock:
                if task_id not in self.tasks:
                    continue
                task = self.tasks[task_id]
                if task.status != TaskStatus.PENDING:
                    continue
                if not all(self.tasks.get(dep_id, Task(status=TaskStatus.COMPLETED, task_id=dep_id, func=lambda: None)).status == TaskStatus.COMPLETED
                           for dep_id in task.dependencies):
                    self._enqueue_task(task)
                    continue
                if not self.resource_monitor.allocate_resources(task.estimated_resources):
                    self._enqueue_task(task)
                    continue
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                asyncio_task = self.loop.create_task(self._run_task(task))
                self.running_tasks[task_id] = asyncio_task
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            time.sleep(1)

async def _run_task(self, task: Task):
    start_time = time.time()
    try:
        result = await asyncio.wait_for(
            self.loop.run_in_executor(self.thread_pool, lambda: task.func(*task.args, **task.kwargs)),
            timeout=task.timeout_seconds
        )
        duration = time.time() - start_time
        with self.task_lock:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = TaskResult(success=True, data=result, duration=duration)
            self.resource_monitor.release_resources(task.estimated_resources)
            if task.task_id in self.running_tasks:
                del self.running_tasks[task_id]
            if self.persist_path:
                self._save_tasks()
        logger.info(f"Completed task {task.task_id} ({task.name}) in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        with self.task_lock:
            task.status = TaskStatus.FAILED if not isinstance(e, asyncio.TimeoutError) else TaskStatus.TIMEOUT
            task.completed_at = datetime.now()
            task.result = TaskResult(success=False, error=str(e), duration=duration)
            self.resource_monitor.release_resources(task.estimated_resources)
            if task.task_id in self.running_tasks:
                del self.running_tasks[task_id]
            if self.persist_path:
                self._save_tasks()
        logger.error(f"Task {task.task_id} ({task.name}) failed: {e}")

def shutdown(self, wait: bool = True):
    self.stop_event.set()
    self.resource_monitor.stop()
    if self.scheduler_thread.is_alive():
        self.scheduler_thread.join(timeout=5.0)
    with self.task_lock:
        for task_id, asyncio_task in list(self.running_tasks.items()):
            asyncio_task.cancel()
    self.thread_pool.shutdown(wait=wait)
    if self.persist_path:
        self._save_tasks()

def _save_tasks(self):
    with open(self.persist_path, 'w') as f:
        json.dump({tid: t.to_dict() for tid, t in self.tasks.items() if t.status != TaskStatus.RUNNING}, f, indent=2)

def _load_tasks(self):
    try:
        with open(self.persist_path, 'r') as f:
            serialized_tasks = json.load(f)
        for task_id, task_dict in serialized_tasks.items():
            if task_dict['status'] in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
                continue
            task = Task(
                task_id=task_id, name=task_dict['name'], func=lambda: None,
                status=TaskStatus[task_dict['status']], priority=TaskPriority[task_dict['priority']],
                created_at=datetime.fromisoformat(task_dict['created_at']),
                timeout_seconds=task_dict['timeout_seconds'], dependencies=task_dict['dependencies']
            )
            if 'completed_at' in task_dict:
                task.completed_at = datetime.fromisoformat(task_dict['completed_at'])
            if 'result' in task_dict:
                task.result = TaskResult(**task_dict['result'])
            self.tasks[task_id] = task
    except Exception as e:
        logger.error(f"Error loading tasks: {e}")

