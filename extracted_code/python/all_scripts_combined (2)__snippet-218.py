    
    with open(dirs["usr/share/applications"] / "echo.desktop", "w") as f:
        f.write(desktop_content)

    # Build .deb
    deb_name = f"echo_{BUILD_CONFIG['version']}_amd64.deb"
    subprocess.check_call(["dpkg-deb", "--build", str(deb_root), str(self.dist_dir / deb_name)])

    print(f"✓ .deb package created: {deb_name}")

def create_linux_appimage(self):
    """Create AppImage"""
    print("📦 Creating AppImage...")

    # This requires appimagetool
    if shutil.which("appimagetool") is None:
        print("⚠️  appimagetool not found. Skipping AppImage creation.")
        print("   Install from: https://appimage.github.io/appimagetool/")
        return

    appdir = self.build_dir / "Echo.AppDir"
    appdir.mkdir(parents=True, exist_ok=True)

    # Copy files
    dist_echo = self.dist_dir / "Echo"
    if dist_echo.exists():
        shutil.copytree(dist_echo, appdir / "usr" / "bin", dirs_exist_ok=True)

    # Create AppRun
    apprun_content = '''#!/bin/bash
