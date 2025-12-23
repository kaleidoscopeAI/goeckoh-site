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

class OptimizedTaskScheduler:
    """Resource-aware task scheduler optimized for CPU environments"""
    
    def __init__(self, 
                max_workers: Optional[int] = None,
                persist_path: Optional[str] = None,
                auto_recovery: bool = True):
        """
        Initialize the task scheduler
        
        Args:
            max_workers: Maximum number of concurrent tasks
            persist_path: Path to persist task state
            auto_recovery: Whether to auto-recover failed tasks
        """
        self.max_workers = max_workers or MAX_WORKERS
        self.persist_path = persist_path or TASK_PERSIST_PATH
        self.auto_recovery = auto_recovery
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.LOW: queue.PriorityQueue(),
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(),
            TaskPriority.CRITICAL: queue.PriorityQueue()
        }
        
        # For tracking running tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # For dependency tracking
        self.dependency_map: Dict[str, List[str]] = {}  # task_id -> dependent task_ids
        
        # Locks
        self.task_lock = threading.Lock()
        
        # Event to stop scheduler
        self.stop_event = threading.Event()
        
        # Resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Thread pools optimized for CPU work
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create event loop
        self.loop = asyncio.new_event_loop()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Load persisted tasks if available
        if self.persist_path and os.path.exists(self.persist_path):
            self._load_tasks()
        
        logger.info(f"Task scheduler initialized with {self.max_workers} workers")
    
    def add_task(self, 
                name: str, 
                func: Callable, 
                args: List = None,
                kwargs: Dict[str, Any] = None,
                priority: TaskPriority = TaskPriority.NORMAL,
                timeout_seconds: int = 3600,
                dependencies: List[str] = None,
                owner: Optional[str] = None,
                metadata: Dict[str, Any] = None,
                estimated_resources: Dict[str, float] = None) -> str:
        """
        Add a task to the scheduler
        
        Args:
            name: Task name
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            timeout_seconds: Timeout in seconds
            dependencies: List of task IDs this task depends on
            owner: User ID or system identifier
            metadata: Additional task metadata
            estimated_resources: Estimated resource requirements (cpu_percent, memory_percent)
            
        Returns:
            Task ID
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Default resources if not provided
        if estimated_resources is None:
            estimated_resources = {
                "cpu_percent": 25.0,  # Default to 25% of a core
                "memory_percent": 10.0  # Default to 10% of system memory
            }
        
        # Create task
        task = Task(
            task_id=task_id,
            name=name,
            func=func,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies or [],
            owner=owner,
            metadata=metadata or {},
            estimated_resources=estimated_resources
        )
        
        # Add to tasks dictionary
        with self.task_lock:
            self.tasks[task_id] = task
            
            # Add to dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.dependency_map:
                    self.dependency_map[dep_id] = []
                self.dependency_map[dep_id].append(task_id)
            
            # Queue task if it has no dependencies
            if not task.dependencies:
                self._enqueue_task(task)
            
            # Persist tasks
            if self.persist_path:
                self._save_tasks()
        
        logger.info(f"Added task {task_id} ({name}) with priority {priority.name}")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: Task ID
            
        Returns:
            Success status
        """
        with self.task_lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for cancellation")
                return False
            
            task = self.tasks[task_id]
            
            # Cancel if pending
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Cancelled pending task {task_id} ({task.name})")
                
                # Also cancel dependent tasks
                if task_id in self.dependency_map:
                    for dep_task_id in self.dependency_map[task_id]:
                        self.cancel_task(dep_task_id)
                
                return True
            
            # Cancel if running
            elif task.status == TaskStatus.RUNNING:
                if task_id in self.running_tasks:
                    # Cancel asyncio task
                    asyncio_task = self.running_tasks[task_id]
                    asyncio_task.cancel()
                    logger.info(f"Cancelled running task {task_id} ({task.name})")
                    
                    # Also cancel dependent tasks
                    if task_id in self.dependency_map:
                        for dep_task_id in self.dependency_map[task_id]:
                            self.cancel_task(dep_task_id)
                    
                    return True
            
            logger.warning(f"Cannot cancel task {task_id} with status {task.status.name}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if not found
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return None
            
            return self.tasks[task_id].status
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get task result
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not found or not completed
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            if task.status != TaskStatus.COMPLETED and task.status != TaskStatus.FAILED:
                return None
            
            return task.result
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task information
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information or None if not found
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return None
            
            return self.tasks[task_id].to_dict()
    
    def list_tasks(self, 
                  status: Optional[TaskStatus] = None, 
                  owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks with optional filters
        
        Args:
            status: Filter by status
            owner: Filter by owner
            
        Returns:
            List of task information dictionaries
        """
        with self.task_lock:
            tasks = []
            
            for task in self.tasks.values():
                if status and task.status != status:
                    continue
                
                if owner and task.owner != owner:
                    continue
                
                tasks.append(task.to_dict())
            
            return tasks
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the scheduler
        
        Args:
            wait: Whether to wait for tasks to complete
        """
        logger.info("Shutting down task scheduler")
        
        # Set stop event
        self.stop_event.set()
        
        # Stop resource monitor
        self.resource_monitor.stop()
        
        # Wait for scheduler thread to exit
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        # Cancel running tasks
        with self.task_lock:
            for task_id, asyncio_task in list(self.running_tasks.items()):
                logger.info(f"Cancelling task {task_id}")
                asyncio_task.cancel()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=wait)
        
        # Save task state
        if self.persist_path:
            self._save_tasks()
    
    def _enqueue_task(self, task: Task):
        """Add task to the appropriate priority queue"""
        queue_item = (-task.priority.value, task.created_at.timestamp(), task.task_id)
        self.task_queues[task.priority].put(queue_item)
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        asyncio.set_event_loop(self.loop)
        logger.info("Task scheduler started")
        
        while not self.stop_event.is_set():
            try:
                # Check for available worker slots
                with self.task_lock:
                    if len(self.running_tasks) >= self.max_workers:
                        # No available workers, wait
                        time.sleep(0.1)
                        continue
                
                # Try to get task from queues by priority
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
                    # No tasks in queue, wait
                    time.sleep(0.1)
                    continue
                
                # Get task
                with self.task_lock:
                    if task_id not in self.tasks:
                        logger.warning(f"Task {task_id} not found in tasks dictionary")
                        continue
                    
                    task = self.tasks[task_id]
                    
                    # Check if task is still pending
                    if task.status != TaskStatus.PENDING:
                        logger.warning(f"Task {task_id} has status {task.status.name}, skipping")
                        continue
                    
                    # Check dependencies
                    all_deps_complete = True
                    for dep_id in task.dependencies:
                        if dep_id not in self.tasks:
                            logger.warning(f"Dependency {dep_id} not found for task {task_id}")
                            all_deps_complete = False
                            break
                        
                        dep_task = self.tasks[dep_id]
                        if dep_task.status != TaskStatus.COMPLETED:
                            all_deps_complete = False
                            break
                    
                    if not all_deps_complete:
                        # Re-queue task
                        self._enqueue_task(task)
                        continue
                    
                    # Check if we have resources
                    if not self.resource_monitor.allocate_resources(task.estimated_resources):
                        logger.info(f"Not enough resources for task {task_id}, re-queueing")
                        self._enqueue_task(task)
                        continue
                    
                    # Start task
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    
                    # Create asyncio task
                    asyncio_task = self.loop.create_task(self._run_task(task))
                    self.running_tasks[task_id] = asyncio_task
                    
                    logger.info(f"Started task {task_id} ({task.name})")
            
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                traceback.print_exc()
                time.sleep(1)  # Avoid tight loop on error
        
        logger.info("Task scheduler stopped")
    
    async def _run_task(self, task: Task):
        """
        Run a task
        
        Args:
            task: Task to run
        """
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_cpu_time = process.cpu_times()
        start_memory = process.memory_info().rss
        
        try:
            # Create task for timeout
            coro = self._execute_task(task)
            
            # Run with timeout
            result = await asyncio.wait_for(coro, timeout=task.timeout_seconds)
            
            # Update task status
            duration = time.time() - start_time
            
            # Calculate resource usage
            process = psutil.Process(os.getpid())
            end_cpu_time = process.cpu_times()
            end_memory = process.memory_info().rss
            
            cpu_usage = (end_cpu_time.user - start_cpu_time.user) / duration * 100
            memory_usage = (end_memory - start_memory) / (psutil.virtual_memory().total) * 100
            
            resource_usage = {
                "cpu_percent": cpu_usage,
                "memory_percent": memory_usage,
                "duration": duration
            }
            
            with self.task_lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = TaskResult(
                    success=True,
                    data=result,
                    duration=duration,
                    resource_usage=resource_usage
                )
                
                # Release resources
                self.resource_monitor.release_resources(task.estimated_resources)
                
                # Check dependents
                if task.task_id in self.dependency_map:
                    for dep_task_id in self.dependency_map[task.task_id]:
                        if dep_task_id in self.tasks:
                            dep_task = self.tasks[dep_task_id]
                            
                            if dep_task.status == TaskStatus.PENDING:
                                # Check if all dependencies are complete
                                all_deps_complete = True
                                for dep_id in dep_task.dependencies:
                                    if dep_id not in self.tasks:
                                        continue
                                    
                                    dep = self.tasks[dep_id]
                                    if dep.status != TaskStatus.COMPLETED:
                                        all_deps_complete = False
                                        break
                                
                                if all_deps_complete:
                                    # Queue dependent task
                                    self._enqueue_task(dep_task)
                
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Persist tasks
                if self.persist_path:
                    self._save_tasks()
            
            logger.info(f"Completed task {task.task_id} ({task.name}) in {duration:.2f}s")
        
        except asyncio.TimeoutError:
            # Task timed out
            duration = time.time() - start_time
            
            with self.task_lock:
                task.status = TaskStatus.TIMEOUT
                task.completed_at = datetime.now()
                task.result = TaskResult(
                    success=False,
                    error=f"Task timed out after {task.timeout_seconds}s",
                    duration=duration
                )
                
                # Release resources
                self.resource_monitor.release_resources(task.estimated_resources)
                
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Retry if needed
                if task.retry_count < task.max_retries and self.auto_recovery:
                    logger.info(f"Scheduling retry #{task.retry_count + 1} for task {task.task_id} ({task.name})")
                    
                    # Create new task for retry
                    new_task = Task(
                        task_id=str(uuid.uuid4()),
                        name=f"{task.name} (retry #{task.retry_count + 1})",
                        func=task.func,
                        args=task.args,
                        kwargs=task.kwargs,
                        priority=task.priority,
                        timeout_seconds=task.timeout_seconds,
                        dependencies=task.dependencies,
                        owner=task.owner,
                        metadata=task.metadata,
                        retry_count=task.retry_count + 1,
                        max_retries=task.max_retries,
                        retry_delay=task.retry_delay,
                        estimated_resources=task.estimated_resources
                    )
                    
                    # Add to tasks
                    self.tasks[new_task.task_id] = new_task
                    
                    # Add to dependencies
                    for dep_id in new_task.dependencies:
                        if dep_id not in self.dependency_map:
                            self.dependency_map[dep_id] = []
                        self.dependency_map[dep_id].append(new_task.task_id)
                    
                    # Schedule retry after delay
                    self.loop.call_later(
                        task.retry_delay,
                        lambda: self._enqueue_task(new_task)
                    )
                else:
                    # Mark dependent tasks as failed
                    if task.task_id in self.dependency_map:
                        for dep_task_id in self.dependency_map[task.task_id]:
                            if dep_task_id in self.tasks:
                                dep_task = self.tasks[dep_task_id]
                                
                                if dep_task.status == TaskStatus.PENDING:
                                    dep_task.status = TaskStatus.FAILED
                                    dep_task.completed_at = datetime.now()
                                    dep_task.result = TaskResult(
                                        success=False,
                                        error=f"Dependency {task.task_id} failed",
                                        duration=0.0
                                    )
                
                # Persist tasks
                if self.persist_path:
                    self._save_tasks()
            
            logger.warning(f"Task {task.task_id} ({task.name}) timed out after {duration:.2f}s")
        
        except asyncio.CancelledError:
            # Task was cancelled
            duration = time.time() - start_time
            
            with self.task_lock:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                task.result = TaskResult(
                    success=False,
                    error="Task was cancelled",
                    duration=duration
                )
                
                # Release resources
                self.resource_monitor.release_resources(task.estimated_resources)
                
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Mark dependent tasks as cancelled
                if task.task_id in self.dependency_map:
                    for dep_task_id in self.dependency_map[task.task_id]:
                        if dep_task_id in self.tasks:
                            dep_task = self.tasks[dep_task_id]
                            
                            if dep_task.status == TaskStatus.PENDING:
                                dep_task.status = TaskStatus.CANCELLED
                                dep_task.completed_at = datetime.now()
                                dep_task.result = TaskResult(
                                    success=False,
                                    error=f"Dependency {task.task_id} was cancelled",
                                    duration=0.0
                                )
                
                # Persist tasks
                if self.persist_path:
                    self._save_tasks()
            
            logger.info(f"Task {task.task_id} ({task.name}) was cancelled after {duration:.2f}s")
        
        except Exception as e:
            # Task failed
            duration = time.time() - start_time
            
            with self.task_lock:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.result = TaskResult(
                    success=False,
                    error=str(e),
                    duration=duration
                )
                
                # Release resources
                self.resource_monitor.release_resources(task.estimated_resources)
                
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Retry if needed
                if task.retry_count < task.max_retries and self.auto_recovery:
                    logger.info(f"Scheduling retry #{task.retry_count + 1} for task {task.task_id} ({task.name})")
                    
                    # Create new task for retry
                    new_task = Task(
                        task_id=str(uuid.uuid4()),
                        name=f"{task.name} (retry #{task.retry_count + 1})",
                        func=task.func,
                        args=task.args,
                        kwargs=task.kwargs,
                        priority=task.priority,
                        timeout_seconds=task.timeout_seconds,
                        dependencies=task.dependencies,
                        owner=task.owner,
                        metadata=task.metadata,
                        retry_count=task.retry_count + 1,
                        max_retries=task.max_retries,
                        retry_delay=task.retry_delay,
                        estimated_resources=task.estimated_resources
                    )
                    
                    # Add to tasks
                    self.tasks[new_task.task_id] = new_task
                    
                    # Add to dependencies
                    for dep_id in new_task.dependencies:
                        if dep_id not in self.dependency_map:
                            self.dependency_map[dep_id] = []
                        self.dependency_map[dep_id].append(new_task.task_id)
                    
                    # Schedule retry after delay
                    self.loop.call_later(
                        task.retry_delay,
                        lambda: self._enqueue_task(new_task)
                    )
                else:
                    # Mark dependent tasks as failed
                    if task.task_id in self.dependency_map:
                        for dep_task_id in self.dependency_map[task.task_id]:
                            if dep_task_id in self.tasks:
                                dep_task = self.tasks[dep_task_id]
                                
                                if dep_task.status == TaskStatus.PENDING:
                                    dep_task.status = TaskStatus.FAILED
                                    dep_task.completed_at = datetime.now()
                                    dep_task.result = TaskResult(
                                        success=False,
                                        error=f"Dependency {task.task_id} failed",
                                        duration=0.0
                                    )
                
                # Persist tasks
                if self.persist_path:
                    self._save_tasks()
            
            logger.error(f"Task {task.task_id} ({task.name}) failed after {duration:.2f}s: {str(e)}")
            traceback.print_exc()
    
    async def _execute_task(self, task: Task) -> Any:
        """
        Execute a task function
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        # Handle coroutine functions
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        
        # Handle regular functions
        # For CPU-bound tasks, we use thread pool to avoid blocking the event loop
        return await self.loop.run_in_executor(
            self.thread_pool,
            lambda: task.func(*task.args, **task.kwargs)
        )
    
    def _save_tasks(self):
        """Save tasks to persistent storage"""
        serializable_tasks = {}
        
        for task_id, task in self.tasks.items():
            # Skip tasks that can't be serialized
            if task.status == TaskStatus.RUNNING:
                continue
            
            task_dict = task.to_dict()
            # Remove function reference
            task_dict.pop('func', None)
            serializable_tasks[task_id] = task_dict
        
        with open(self.persist_path, 'w') as f:
            json.dump(serializable_tasks, f, indent=2)
    
    def _load_tasks(self):
        """Load tasks from persistent storage"""
        try:
            with open(self.persist_path, 'r') as f:
                serialized_tasks = json.load(f)
            
            for task_id, task_dict in serialized_tasks.items():
                # Skip tasks that need function reference
                if task_dict.get('status') in [TaskStatus.PENDING.name, TaskStatus.RUNNING.name]:
                    continue
                
                # Create task
                task = Task(
                    task_id=task_id,
                    name=task_dict['name'],
                    func=None,  # Can't deserialize functions
                    status=TaskStatus[task_dict['status']],
                    priority=TaskPriority[task_dict['priority']],
                    created_at=datetime.fromisoformat(task_dict['created_at']),
                    timeout_seconds=task_dict['timeout_seconds'],
                    retry_count=task_dict['retry_count'],
                    max_retries=task_dict['max_retries'],
                    retry_delay=task_dict['retry_delay'],
                    dependencies=task_dict['dependencies'],
                    owner=task_dict['owner'],
                    metadata=task_dict['metadata'],
                    estimated_resources=task_dict.get('estimated_resources', {})
                )
                
                # Add started_at and completed_at if available
                if 'started_at' in task_dict:
                    task.started_at = datetime.fromisoformat(task_dict['started_at'])
                
                if 'completed_at' in task_dict:
                    task.completed_at = datetime.fromisoformat(task_dict['completed_at'])
                
                # Add result if available
                if 'result' in task_dict:
                    result_dict = task_dict['result']
                    task.result = TaskResult(
                        success=result_dict['success'],
                        error=result_dict.get('error'),
                        duration=result_dict['duration'],
                        resource_usage=result_dict.get('resource_usage', {})
                    )
                
                # Add to tasks
                self.tasks[task_id] = task
                
                # Add to dependencies
                for dep_id in task.dependencies:
                    if dep_id not in self.dependency_map:
                        self.dependency_map[dep_id] = []
                    self.dependency_map[dep_id].append(task_id)
            
            logger.info(f"Loaded {len(self.tasks)} tasks from {self.persist_path}")
        
        except Exception as e:
            logger.error(f"Error loading tasks from {self.persist_path}: {str(e)}")
            traceback.print_exc()

