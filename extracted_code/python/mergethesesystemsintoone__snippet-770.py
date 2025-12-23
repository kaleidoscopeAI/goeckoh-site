def __init__(self, work_dir: str = "unravel_ai_workdir"):
    self.work_dir = Path(work_dir)
    self.work_dir.mkdir(exist_ok=True, parents=True)
    self.scheduler = OptimizedTaskScheduler(persist_path=str(self.work_dir / "tasks.json"))
    self.analysis_dir = self.work_dir / "analysis"
    self.analysis_dir.mkdir(exist_ok=True)

def create_unravel_tasks(self, input_directory: str, target_language: Optional[str] = None, extra_tasks: List[str] = None) -> List[str]:
    session_id = str(uuid.uuid4())[:8]
    session_dir = self.analysis_dir / session_id
    session_dir.mkdir(exist_ok=True)
    task_ids = []
    extra_tasks = extra_tasks or []

    def setup_task():
        os.makedirs(session_dir, exist_ok=True)
        return "Setup complete"

    def analyze_task():
        time.sleep(5)  # Simulate analysis
        return f"Analyzed {input_directory}"

    task_ids.append(self.scheduler.add_task("setup", setup_task, priority=TaskPriority.HIGH))
    task_ids.append(self.scheduler.add_task("analyze", analyze_task, dependencies=[task_ids[0]]))
    if "security" in extra_tasks:
        task_ids.append(self.scheduler.add_task("security", lambda: "Security check", dependencies=[task_ids[1]]))
    if target_language:
        task_ids.append(self.scheduler.add_task("reconstruct", lambda: f"Reconstructed to {target_language}", dependencies=[task_ids[1]]))
    return task_ids

def run(self):
    # The scheduler runs in its own thread; just wait for tasks to complete or implement a specific run logic if needed
    while len(self.scheduler.running_tasks) > 0 or any(not q.empty() for q in self.scheduler.task_queues.values()):
        time.sleep(1)

