def __init__(self, work_dir: str = "unravel_ai_workdir"):
    self.work_dir = Path(work_dir)
    self.work_dir.mkdir(exist_ok=True, parents=True)
    self.scheduler = OptimizedTaskScheduler(persist_path=str(self.work_dir / "tasks.json"))
    self.llm_processor = LLMProcessor()
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

    def analyze_code_task(input_dir=session_dir):
        code_texts = []
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith(('.py', '.cpp', '.java', '.js')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        code_texts.append(f.read())
        structure = [self.llm_processor.analyze_text_structure(text) for text in code_texts]
        sentiment = self.llm_processor.classify_text(code_texts)
        return {"structure": structure, "sentiment": sentiment}

    def summarize_code_task(input_dir=session_dir):
        code_texts = []
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith(('.py', '.cpp', '.java', '.js')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        code_texts.append(f.read())
        summaries = self.llm_processor.summarize_text(code_texts)
        with open(session_dir / "summaries.json", 'w') as f:
            json.dump(summaries, f, indent=2)
        return summaries

    task_ids.append(self.scheduler.add_task("setup", setup_task, priority=TaskPriority.HIGH))
    task_ids.append(self.scheduler.add_task("analyze_code", analyze_code_task, dependencies=[task_ids[0]]))
    task_ids.append(self.scheduler.add_task("summarize_code", summarize_code_task, dependencies=[task_ids[1]]))
    if "security" in extra_tasks:
        task_ids.append(self.scheduler.add_task("security", lambda: "Security check", dependencies=[task_ids[1]]))
    if target_language:
        task_ids.append(self.scheduler.add_task("reconstruct", lambda: f"Reconstructed to {target_language}", dependencies=[task_ids[1]]))
    return task_ids

def run(self):
    while len(self.scheduler.running_tasks) > 0 or any(not q.empty() for q in self.scheduler.task_queues.values()):
        time.sleep(1)

