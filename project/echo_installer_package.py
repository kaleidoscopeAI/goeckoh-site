#!/usr/bin/env python3
"""
Echo v4.0 Crystalline Heart - Complete Build & Package System
==============================================================

This script creates production-ready installers for:
- Windows (.exe with installer)
- macOS (.app bundle + .dmg)
- Linux (.deb, .AppImage)

Usage:
    python build_and_package.py --platform windows
    python build_and_package.py --platform macos
    python build_and_package.py --platform linux
    python build_and_package.py --platform all
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import json

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

BUILD_CONFIG = {
    "app_name": "Echo",
    "version": "4.0.0",
    "author": "Echo Development Team",
    "description": "Autism-tuned speech companion with crystalline emotional lattice",
    "identifier": "com.echo.companion",
    "license": "Proprietary",
    "python_version": "3.11",
    "icon_name": "echo_icon"
}

# ============================================================================
# FILE: installer_bootstrap.py
# This gets bundled into the installer to handle first-run setup
# ============================================================================

INSTALLER_BOOTSTRAP = '''
"""Echo First-Run Setup"""
import os
import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog
import threading

class EchoSetupWizard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Echo v4.0 - First-Time Setup")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        self.voice_path = None
        self.ollama_enabled = tk.BooleanVar(value=False)
        
        self.create_ui()
    
    def create_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#4f46e5", height=80)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header,
            text="🎙️ Echo v4.0",
            bg="#4f46e5",
            fg="white",
            font=("Helvetica", 24, "bold")
        )
        title.pack(pady=20)
        
        # Main content
        content = tk.Frame(self.root, padx=30, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Welcome text
        welcome = tk.Label(
            content,
            text="Welcome to Jackson's Crystalline Speech Companion",
            font=("Helvetica", 14),
            wraplength=500
        )
        welcome.pack(pady=(0, 20))
        
        # Step 1: Voice sample
        step1_frame = tk.LabelFrame(content, text="Step 1: Voice Sample", padx=10, pady=10)
        step1_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            step1_frame,
            text="Record 10-30 seconds of the child speaking naturally.\\n"
                 "This will be used to clone their voice.",
            wraplength=500,
            justify=tk.LEFT
        ).pack(anchor=tk.W)
        
        tk.Button(
            step1_frame,
            text="📁 Select Voice Recording (WAV file)",
            command=self.select_voice_file,
            bg="#4f46e5",
            fg="white",
            font=("Helvetica", 11, "bold"),
            padx=20,
            pady=10
        ).pack(pady=10)
        
        self.voice_label = tk.Label(step1_frame, text="No file selected", fg="gray")
        self.voice_label.pack()
        
        # Step 2: LLM (optional)
        step2_frame = tk.LabelFrame(content, text="Step 2: Inner Voice AI (Optional)", padx=10, pady=10)
        step2_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            step2_frame,
            text="Enable local AI for gentle inner voice phrases?\\n"
                 "Requires Ollama to be installed separately.",
            wraplength=500,
            justify=tk.LEFT
        ).pack(anchor=tk.W)
        
        tk.Checkbutton(
            step2_frame,
            text="Enable AI Inner Voice",
            variable=self.ollama_enabled,
            font=("Helvetica", 10)
        ).pack(anchor=tk.W, pady=5)
        
        # Start button
        tk.Button(
            content,
            text="🚀 Start Echo",
            command=self.start_echo,
            bg="#22c55e",
            fg="white",
            font=("Helvetica", 14, "bold"),
            padx=40,
            pady=15
        ).pack(pady=20)
    
    def select_voice_file(self):
        path = filedialog.askopenfilename(
            title="Select Voice Recording",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.voice_path = path
            self.voice_label.config(text=f"✓ {Path(path).name}", fg="green")
    
    def start_echo(self):
        if not self.voice_path:
            messagebox.showerror("Error", "Please select a voice recording first!")
            return
        
        # Copy voice file to config location
        config_dir = Path.home() / ".echo_companion" / "voices"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = config_dir / "child_ref.wav"
        shutil.copy2(self.voice_path, target_path)
        
        # Save config
        config = {
            "voice_sample": str(target_path),
            "llm_enabled": self.ollama_enabled.get(),
            "first_run_complete": True
        }
        
        config_file = Path.home() / ".echo_companion" / "setup.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        messagebox.showinfo(
            "Setup Complete",
            "Echo is ready!\\n\\nThe application will now start.\\n\\n"
            "Open your browser to: http://localhost:8000/static/index.html"
        )
        
        self.root.destroy()
        
        # Start the server
        self.launch_server()
    
    def launch_server(self):
        import subprocess
        
        # Get the installation directory
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys._MEIPASS)
        else:
            app_dir = Path(__file__).parent
        
        # Start uvicorn server
        subprocess.Popen([
            sys.executable,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000"
        ], cwd=app_dir)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    wizard = EchoSetupWizard()
    wizard.run()
'''

# ============================================================================
# FILE: build_and_package.py (main build script)
# ============================================================================

class EchoBuilder:
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
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('frontend', 'frontend'),
        ('echo_core', 'echo_core'),
        ('config.py', '.'),
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        'uvicorn',
        'fastapi',
        'sounddevice',
        'soundfile',
        'faster_whisper',
        'TTS',
        'language_tool_python',
        'torch',
        'torchaudio',
        'numpy',
        'scipy',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Echo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/echo_icon.ico' if os.path.exists('assets/echo_icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Echo',
)

{"" if self.platform != "macos" else """
app = BUNDLE(
    coll,
    name='Echo.app',
    icon='assets/echo_icon.icns',
    bundle_identifier='com.echo.companion',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '4.0.0',
        'NSMicrophoneUsageDescription': 'Echo needs microphone access to listen to speech.',
    },
)
"""}
'''
        
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
[Setup]
AppName={BUILD_CONFIG["app_name"]}
AppVersion={BUILD_CONFIG["version"]}
AppPublisher={BUILD_CONFIG["author"]}
DefaultDirName={{autopf}}\\Echo
DefaultGroupName=Echo
OutputDir=dist
OutputBaseFilename=Echo-v{BUILD_CONFIG["version"]}-Windows-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
UninstallDisplayIcon={{app}}\\Echo.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "dist\\Echo\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\Echo"; Filename: "{{app}}\\Echo.exe"
Name: "{{commondesktop}}\\Echo"; Filename: "{{app}}\\Echo.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\Echo.exe"; Description: "Launch Echo"; Flags: nowait postinstall skipifsilent
'''
        
        iss_file = self.project_root / "echo_installer.iss"
        with open(iss_file, "w") as f:
            f.write(iss_content)
        
        # Run Inno Setup
        subprocess.check_call([str(inno_path), str(iss_file)])
        print("✓ Windows installer created")
    
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
Version: {BUILD_CONFIG["version"]}
Section: utils
Priority: optional
Architecture: amd64
Maintainer: {BUILD_CONFIG["author"]}
Description: {BUILD_CONFIG["description"]}
 Autism-tuned speech companion with crystalline emotional lattice.
 Provides real-time speech mirroring in the child's own voice.
'''
        
        with open(dirs["DEBIAN"] / "control", "w") as f:
            f.write(control_content)
        
        # Create .desktop file
        desktop_content = f'''[Desktop Entry]
Type=Application
Name=Echo
Comment={BUILD_CONFIG["description"]}
Exec=/opt/echo/Echo
Icon=echo
Terminal=false
Categories=Education;Accessibility;
'''
        
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
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/Echo" "$@"
'''
        
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

# ============================================================================
# DEPLOYMENT GUIDE (Markdown)
# ============================================================================

DEPLOYMENT_GUIDE = '''
# Echo v4.0 Deployment Guide

## Prerequisites

### All Platforms
- Python 3.10 or 3.11
- 4GB RAM minimum (8GB recommended)
- Microphone access
- 2GB free disk space

### Platform-Specific
- **Windows**: Windows 10/11, Inno Setup (for installer creation)
- **macOS**: macOS 11+, Xcode Command Line Tools
- **Linux**: Ubuntu 20.04+ or equivalent

## Building from Source

### 1. Clone/Extract Repository
```bash
cd echo_companion
```

### 2. Install Build Dependencies
```bash
pip install -r requirements.txt
pip install pyinstaller
```

### 3. Run Build Script

**Windows:**
```bash
python build_and_package.py --platform windows --clean
```

**macOS:**
```bash
python build_and_package.py --platform macos --clean
```

**Linux:**
```bash
python build_and_package.py --platform linux --clean
```

**All Platforms:**
```bash
python build_and_package.py --platform all --clean
```

### 4. Locate Installers
Check the `dist/` folder for:
- Windows: `Echo-v4.0.0-Windows-Setup.exe`
- macOS: `Echo-v4.0.0-macOS.dmg`
- Linux: `echo_4.0.0_amd64.deb` and `Echo-v4.0.0-x86_64.AppImage`

## Installation

### Windows
1. Run `Echo-v4.0.0-Windows-Setup.exe`
2. Follow installer prompts
3. Launch from Start Menu or Desktop shortcut

### macOS
1. Open `Echo-v4.0.0-macOS.dmg`
2. Drag Echo.app to Applications folder
3. First launch: Right-click → Open (to bypass Gatekeeper)
4. Grant microphone permissions when prompted

### Linux (Debian/Ubuntu)
```bash
sudo dpkg -i echo_4.0.0_amd64.deb
sudo apt-get install -f  # Install dependencies
```

Or use AppImage:
```bash
chmod +x Echo-v4.0.0-x86_64.AppImage
./Echo-v4.0.0-x86_64.AppImage
```

## First-Time Setup

1. **Launch Echo** - The setup wizard will appear
2. **Record Voice Sample** - Record 10-30 seconds of the child speaking naturally
3. **Select the WAV file** in the setup wizard
4. **(Optional) Enable AI Inner Voice** - Requires Ollama installed separately
5. **Click "Start Echo"**
6. **Browser opens** to `http://localhost:8000/static/index.html`

## Configuration

### Voice Sample Location
```
Windows: C:\\Users\\<username>\\.echo_companion\\voices\\child_ref.wav
macOS: /Users/<username>/.echo_companion/voices/child_ref.wav
Linux: /home/<username>/.echo_companion/voices/child_ref.wav
```

### Log Files
```
Windows: C:\\Users\\<username>\\.echo_companion\\logs\\
macOS: /Users/<username>/.echo_companion/logs/
Linux: /home/<username>/.echo_companion/logs/
```

### Optional: Ollama Setup (for AI Inner Voice)
1. Download Ollama from https://ollama.ai
2. Install and run: `ollama pull deepseek-r1:8b`
3. Enable in Echo settings

## Usage

### For the Child (Jackson's View)
- Click the microphone button to start listening
- Speak naturally with pauses as needed
- Echo waits 1.2 seconds of silence before processing
- Hear your corrected speech in your own voice
- All phrases use "I/me/my" language

### For the Parent (Molly's View)
- Monitor emotional arousal and valence in real-time
- See behavior flags (anxiety, perseveration)
- Review session history
- Track progress over days/weeks

## Troubleshooting

### "Microphone not detected"
- Windows: Settings → Privacy → Microphone → Allow apps
- macOS: System Preferences → Security & Privacy → Microphone
- Linux: Check `pavucontrol` or `alsamixer`

### "Voice cloning fails"
- Ensure WAV file is 16kHz or 24kHz mono
- Recording should be 10-30 seconds
- Clear speech with minimal background noise

### "LLM inner voice not working"
- Check Ollama is running: `ollama serve`
- Verify model is pulled: `ollama list`
- Check connection: `curl http://localhost:11434`

### "High CPU usage"
- Normal during speech processing
- Reduce `emotion.num_nodes` in config.py to 64
- Use faster STT model: `stt.model_size = "tiny"`

## Development Mode

Run without building:
```bash
# Terminal 1: Start backend
uvicorn server:app --reload

# Terminal 2 (optional): Monitor logs
tail -f ~/.echo_companion/logs/*.log
```

Open browser to `http://localhost:8000/static/index.html`

## Uninstall

### Windows
Control Panel → Programs → Uninstall Echo

### macOS
Drag Echo.app to Trash
Remove config: `rm -rf ~/.echo_companion`

### Linux
```bash
sudo apt-get remove echo
rm -rf ~/.echo_companion
```

## Support & Feedback

This is a specialized assistive technology system designed for autistic children.
For questions about the emotional lattice, first-person transformation, or 
clinical applications, refer to the patent documentation.

## Security & Privacy

- **100% offline operation** - No data leaves the device
- **HIPAA-ready architecture** - All data stored locally
- **No telemetry** - Zero usage tracking
- **Open logs** - All session data in readable JSON format

## License

Proprietary - Echo v4.0 Crystalline Heart
© 2025 Echo Development Team
'''

# ============================================================================
# Main execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build and package Echo v4.0 for distribution"
    )
    parser.add_argument(
        "--platform",
        choices=["windows", "macos", "linux", "all"],
        required=True,
        help="Target platform(s) to build for"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directories before building"
    )
    
    args = parser.parse_args()
    
    # Save deployment guide
    guide_path = Path("DEPLOYMENT_GUIDE.md")
    with open(guide_path, "w") as f:
        f.write(DEPLOYMENT_GUIDE)
    print(f"📖 Deployment guide saved to: {guide_path}")
    
    # Save installer bootstrap
    bootstrap_path = Path("installer_bootstrap.py")
    with open(bootstrap_path, "w") as f:
        f.write(INSTALLER_BOOTSTRAP)
    print(f"📄 Installer bootstrap saved to: {bootstrap_path}")
    
    # Build for requested platforms
    platforms = ["windows", "macos", "linux"] if args.platform == "all" else [args.platform]
    
    for platform in platforms:
        builder = EchoBuilder(platform, clean=args.clean)
        builder.build()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Echo v4.0 Crystalline Heart - Build & Package Tool      ║
║                                                              ║
║  Complete installer and deployment system for Windows,      ║
║  macOS, and Linux platforms.                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

This script will create:
  • Standalone executables (PyInstaller)
  • Platform-specific installers
  • First-run setup wizard
  • Complete deployment documentation

Usage examples:
  python build_and_package.py --platform windows --clean
  python build_and_package.py --platform all

""")
    
    # For artifact demonstration, show the structure
    print("\n📦 Package Components:\n")
    print("1. build_and_package.py - Main build orchestration script")
    print("2. installer_bootstrap.py - First-run setup wizard")
    print("3. echo.spec - PyInstaller configuration")
    print("4. DEPLOYMENT_GUIDE.md - Complete deployment documentation")
    print("\n✓ Save this file as 'build_and_package.py' and run it!")
    print("\nExample: python build_and_package.py --platform windows --clean")
