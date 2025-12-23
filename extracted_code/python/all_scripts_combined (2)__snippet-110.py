    def create_macos_dmg(self):
        """Create macOS DMG installer"""
        print("📦 Creating macOS DMG...")
        
        app_path = self.dist_dir / "Echo.app"
        if not app_path.exists():
            print("❌ Echo.app not found!")
            return
        
        dmg_name = f"Echo-v{BUILD_CONFIG['version']}-macOS.dmg"
        dmg_path = self.dist_dir / dmg_name
        
        # Create DMG using hdiutil
        cmd = [
            "hdiutil", "create",
            "-volname", "Echo",
            "-srcfolder", str(app_path),
            "-ov",
            "-format", "UDZO",
            str(dmg_path)
        ]
        
        subprocess.check_call(cmd)
        print(f"✓ DMG created: {dmg_path}")
    
    def create_linux_deb(self):
        """Create Debian package"""
        print("📦 Creating .deb package...")
        
        deb_root = self.build_dir / "deb"
        deb_root.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        dirs = {
            "DEBIAN": deb_root / "DEBIAN",
            "opt": deb_root / "opt" / "echo",
            "usr/share/applications": deb_root / "usr" / "share" / "applications",
            "usr/share/pixmaps": deb_root / "usr" / "share" / "pixmaps",
        }
        
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Copy application files
        dist_echo = self.dist_dir / "Echo"
        if dist_echo.exists():
            shutil.copytree(dist_echo, dirs["opt"], dirs_exist_ok=True)
        
        # Create control file
        control_content = f'''Package: echo
