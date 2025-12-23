    
    apprun_path = appdir / "AppRun"
    with open(apprun_path, "w") as f:
        f.write(apprun_content)
    apprun_path.chmod(0o755)

    # Build AppImage
    appimage_name = f"Echo-v{BUILD_CONFIG['version']}-x86_64.AppImage"
    subprocess.check_call([
        "appimagetool",
        str(appdir),
        str(self.dist_dir / appimage_name)
    ])

    print(f"✓ AppImage created: {appimage_name}")

def build(self):
    """Main build orchestration"""
    print(f"\n{'='*60}")
    print(f"Building Echo v{BUILD_CONFIG['version']} for {self.platform}")
    print(f"{'='*60}\n")

    if self.clean:
        self.clean_build_dirs()

    self.check_dependencies()
    self.build_executable()

    if self.platform == "windows":
        self.create_windows_installer()
    elif self.platform == "macos":
        self.create_macos_dmg()
    elif self.platform == "linux":
        self.create_linux_deb()
        self.create_linux_appimage()

    print(f"\n✅ Build complete! Check the 'dist' folder.\n")

