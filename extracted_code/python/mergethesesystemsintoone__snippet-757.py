def create_unravel_tasks(self, input_directory: str, target_language: Optional[str] = None, 
                        extra_tasks: List[str] = None) -> List[str]:
    """Create a set of tasks for processing a codebase with UnravelAI"""
    logger.info(f"Creating UnravelAI tasks for {input_directory}")

    # Ensure input directory exists
    if not os.path.exists(input_directory):
        raise ValueError(f"Input directory not found: {input_directory}")

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    session_dir = self.analysis_dir / session_id
    session_dir.mkdir(exist_ok=True)

    # Task IDs for tracking
    task_ids = []

    # 1. Create setup task
    setup_task = TaskConfig(
        task_name=f"unravel_setup_{session_id}",
        description="Setup and preparation for UnravelAI analysis",
        priority=10,
        command=["python", "-c", f"import os; os.makedirs('{session_dir}', exist_ok=True); print('Setup complete')"],
        status="pending"
    )
    task_ids.append(self.add_task(setup_task))

    # ... (rest of the function remains the same)

    logger.info(f"Created {len(task_ids)} UnravelAI tasks for session {session_id}")
    return task_ids
