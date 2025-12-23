    
    spec_file = self.project_root / "echo.spec"
    with open(spec_file, "w") as f:
        f.write(spec_content)

    print(f"✓ Spec file created: {spec_file}")
    return spec_file

def build_executable(self):
    """Run PyInstaller to build executable"""
    print("🏗️  Building executable with PyInstaller...")

    spec_file = self.create_spec_file()

    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ]

    subprocess.check_call(cmd)
    print("✓ Executable built successfully")

def create_windows_installer(self):
    """Create Windows installer using Inno Setup"""
    print("📦 Creating Windows installer...")

    # Check for Inno Setup
    inno_path = Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe")
    if not inno_path.exists():
        print("⚠️  Inno Setup not found. Skipping installer creation.")
        print("   Download from: https://jrsoftware.org/isdl.php")
        return

    # Create Inno Setup script
    iss_content = f'''
