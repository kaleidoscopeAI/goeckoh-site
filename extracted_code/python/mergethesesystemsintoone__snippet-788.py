"""Analyzes a system to gather information needed for upgrading"""

def __init__(self):
    """Initialize system analyzer"""
    self.language_detector = LanguageDetector()
    self.excluded_dirs = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__", 
        "venv", "env", ".env", ".venv", "dist", "build"
    }
    self.excluded_files = {
        ".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore"
    }

def analyze_system(self, path: str) -> SystemInfo:
    """
    Analyze a system to gather information

    Args:
        path: Path to the system root directory

    Returns:
        System information
    """
    logger.info(f"Analyzing system at {path}")

    # Initialize system info
    system_info = SystemInfo(
        root_path=path,
        system_type=SystemType.UNKNOWN,
        primary_language=LanguageType.UNKNOWN
    )

    # Check if path exists
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    # Count languages for later determining primary language
    language_counts = {}

    # Walk through the directory tree
    total_size = 0
    file_count = 0

    for root, dirs, files in os.walk(path, topdown=True):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

        # Process each file
        for file in files:
            if file in self.excluded_files:
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, path)





