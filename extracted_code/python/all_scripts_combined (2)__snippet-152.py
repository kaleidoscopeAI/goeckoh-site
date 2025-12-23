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

