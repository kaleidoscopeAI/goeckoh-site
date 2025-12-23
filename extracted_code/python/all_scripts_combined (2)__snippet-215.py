def __init__(self, platform, clean=False):
    self.platform = platform
    self.clean = clean
    self.project_root = Path.cwd()
    self.build_dir = self.project_root / "build"
    self.dist_dir = self.project_root / "dist"

def clean_build_dirs(self):
    """Remove old build artifacts"""
    print("🧹 Cleaning build directories...")
    for dir_path in [self.build_dir, self.dist_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    print("✓ Clean complete")

def check_dependencies(self):
    """Verify all required tools are installed"""
    print("🔍 Checking dependencies...")

    required = {
        "python": "Python 3.10+",
        "pip": "pip",
    }

    for cmd, name in required.items():
        if shutil.which(cmd) is None:
            print(f"❌ {name} not found!")
            sys.exit(1)

    # Check PyInstaller
    try:
        import PyInstaller
        print("✓ PyInstaller found")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    print("✓ All dependencies satisfied")

def create_spec_file(self):
    """Generate PyInstaller spec file"""
    print("📝 Creating PyInstaller spec file...")

    spec_content = f'''
