work_dir: str = "unravel_ai_workdir"
max_concurrent_tasks: int = 4
log_level: str = "INFO"
show_progress: bool = True
abort_on_fail: bool = False
save_results: bool = True
max_memory_usage: float = 0.8
max_cpu_usage: float = 0.8
auto_restart: bool = True
retry_delay: int = 5
backup_interval: int = 300

def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

