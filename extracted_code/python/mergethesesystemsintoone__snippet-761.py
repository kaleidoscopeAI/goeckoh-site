def __init__(self, config: TaskManagerConfig):
    super().__init__(config)
    self.uploads_dir = self.work_dir / "uploads"
    self.uploads_dir.mkdir(exist_ok=True)
    self.analysis_dir = self.work_dir / "analysis"
    self.analysis_dir.mkdir(exist_ok=True)
    self.reconstructed_dir = self.work_dir / "reconstructed"
    self.reconstructed_dir.mkdir(exist_ok=True)
    self.viz_dir = self.work_dir / "visualizations"
    self.viz_dir.mkdir(exist_ok=True)

    # New: Cache for analysis results
    self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    logger.info("UnravelAI Task Manager initialized")

def create_unravel_tasks(self, input_directory: str, target_language: Optional[str] = None, 
                        extra_tasks: List[str] = None) -> List[str]:
    """Create a set of tasks for processing a codebase with UnravelAI"""
    logger.info(f"Creating UnravelAI tasks for {input_directory}")

    if not os.path.exists(input_directory):
        raise ValueError(f"Input directory not found: {input_directory}")

    session_id = str(uuid.uuid4())[:8]
    session_dir = self.analysis_dir / session_id
    session_dir.mkdir(exist_ok=True)

    task_ids = []
    extra_tasks = extra_tasks or []

    # Setup task
    setup_task = TaskConfig(
        task_name=f"unravel_setup_{session_id}",
        description="Setup and preparation for UnravelAI analysis",
        priority=10,
        command=["python", "-c", f"import os; os.makedirs('{session_dir}', exist_ok=True); print('Setup complete')"],
        status="pending",
        tags=["setup", "unravel"]
    )
    task_ids.append(self.add_task(setup_task))

    # File analysis task
    analysis_task = TaskConfig(
        task_name=f"unravel_analyze_{session_id}",
        description="Analyze codebase files and build quantum network",
        priority=8,
        dependencies=[task_ids[0]],
        command=[
            sys.executable,
            "-m",
            "unravel_ai_core",  # Assuming this module exists
            "--input",
            input_directory,
            "--output",
            str(session_dir),
            "--analyze-only"
        ],
        status="pending",
        timeout=3600,  # 1 hour timeout
        tags=["analysis", "unravel"]
    )
    task_ids.append(self.add_task(analysis_task))

    # Pattern detection task
    patterns_task = TaskConfig(
        task_name=f"unravel_patterns_{session_id}",
        description="Detect emergent patterns in code",
        priority=6,
        dependencies=[task_ids[1]],
        command=[
            sys.executable,
            "-m",
            "unravel_ai_core",
            "--session",
            session_id,
            "--detect-patterns"
        ],
        status="pending",
        tags=["patterns", "unravel"]
    )
    task_ids.append(self.add_task(patterns_task))

    # Security analysis task
    if "security_analysis" in extra_tasks:
        security_task = TaskConfig(
            task_name=f"unravel_security_{session_id}",
            description="Perform security vulnerability analysis",
            priority=7,
            dependencies=[task_ids[1]],
            command=[
                sys.executable,
                "-m",
                "unravel_ai_core",
                "--session",
                session_id,
                "--security-analysis"
            ],
            output_file=f"security_analysis_{session_id}.json",
            status="pending",
            tags=["security", "unravel"]
        )
        task_ids.append(self.add_task(security_task))

    # Code optimization task
    if "code_optimization" in extra_tasks:
        optimize_task = TaskConfig(
            task_name=f"unravel_optimize_{session_id}",
            description="Perform code optimization analysis",
            priority=5,
            dependencies=[task_ids[1]],
            command=[
                sys.executable,
                "-m",
                "unravel_ai_core",
                "--session",
                session_id,
                "--optimize-code"
            ],
            output_file=f"optimization_{session_id}.json",
            status="pending",
            tags=["optimization", "unravel"]
        )
        task_ids.append(self.add_task(optimize_task))

    # Visualization task
    viz_task = TaskConfig(
        task_name=f"unravel_visualize_{session_id}",
        description="Generate quantum network visualization",
        priority=4,
        dependencies=[task_ids[2]],
        command=[
            sys.executable,
            "-m",
            "unravel_ai_core",
            "--session",
            session_id,
            "--visualize"
        ],
        output_file=f"visualization_{session_id}.log",
        status="pending",
        tags=["visualization", "unravel"]
    )
    task_ids.append(self.add_task(viz_task))

    # Reconstruction task
    if target_language:
        recon_task = TaskConfig(
            task_name=f"unravel_reconstruct_{session_id}",
            description=f"Reconstruct codebase in {target_language}",
            priority=3,
            dependencies=[task_ids[2]],
            command=[
                sys.executable,
                "-m",
                "unravel_ai_core",
                "--session",
                session_id,
                "--target",
                target_language,
                "--reconstruct"
            ],
            output_file=f"reconstruction_{session_id}.log",
            status="pending",
            tags=["reconstruction", "unravel"]
        )
        task_ids.append(self.add_task(recon_task))

    # Report generation task
    report_task = TaskConfig(
        task_name=f"unravel_report_{session_id}",
        description="Generate comprehensive analysis report",
        priority=2,
        dependencies=task_ids[1:],
        command=[
            sys.executable,
            "-m",
            "unravel_ai_core",
            "--session",
            session_id,
            "--generate-report"
        ],
        output_file=f"report_{session_id}.json",
        status="pending",
        tags=["report", "unravel"]
    )
    task_ids.append(self.add_task(report_task))

    logger.info(f"Created {len(task_ids)} UnravelAI tasks for session {session_id}")
    return task_ids

# Enhanced analyze_results with caching
def analyze_results(self, session_id: str) -> Dict[str, Any]:
    """Analyze results from a completed session with caching"""
    if session_id in self.analysis_cache:
        logger.info(f"Returning cached analysis for session {session_id}")
        return self.analysis_cache[session_id]

    session_dir = self.analysis_dir / session_id
    if not session_dir.exists():
        raise ValueError(f"Session directory not found: {session_dir}")

    analysis_file = session_dir / "analysis.json"
    if not analysis_file.exists():
        raise ValueError(f"Analysis file not found: {analysis_file}")

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    metrics = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "file_count": analysis.get("file_count", 0),
        "emergent_intelligence_score": analysis.get("emergent_properties", {}).get("emergent_intelligence_score", 0.0),
        "pattern_count": len(analysis.get("emergent_properties", {}).get("patterns", [])),
        "coherence": analysis.get("coherence", 0.0),
        "complexity_score": self._calculate_complexity_score(analysis),
        "maintainability_score": self._calculate_maintainability_score(analysis),
        "security_score": self._calculate_security_score(analysis),
        "optimization_potential": self._calculate_optimization_potential(analysis)
    }

    summary = self._generate_analysis_summary(metrics, analysis)
    result = {
        "metrics": metrics,
        "summary": summary,
        "analysis": analysis,
        "recommendations": self._generate_recommendations(analysis)
    }

    result_file = session_dir / "analysis_summary.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    self.analysis_cache[session_id] = result
    return result

# Rest of UnravelAITaskManager methods remain largely unchanged, just adding minor enhancements
def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
    score = 0.5
    if "network_analysis" in analysis:
        network = analysis["network_analysis"]
        metrics = network.get("metrics", {})
        if "connected_components" in metrics:
            components = metrics["connected_components"]
            score += min(0.2, components * 0.05)
        if "max_degree_centrality" in metrics:
            centrality = metrics["max_degree_centrality"]
            score += centrality * 0.3
        if "average_clustering" in metrics:
            clustering = metrics["average_clustering"]
            score += clustering * 0.1
        if "has_cycles" in metrics and metrics["has_cycles"]:
            score += 0.1
    return min(1.0, score)

# ... (other calculation methods remain similar)

def generate_interactive_visualization(self, session_id: str) -> Optional[str]:
    # ... (keeping existing implementation)
    pass

def visualize_analysis_results(self, session_id: str) -> None:
    # ... (keeping existing implementation)
    pass

