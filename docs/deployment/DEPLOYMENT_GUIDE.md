# Deployment Guide - Goeckoh System

## 📦 Packaging Options

The system can be packaged in multiple formats for deployment:

### 1. Source Package (Tar/Zip)
**Best for**: Custom installations, development, Linux distributions

```bash
./package_deployment.sh
```

Creates:
- `dist/goeckoh-system-1.0.0.tar.gz` - Source package
- `dist/goeckoh-system-1.0.0.zip` - Alternative format

**Installation:**
```bash
tar -xzf goeckoh-system-1.0.0.tar.gz
cd goeckoh-system-1.0.0
./install.sh
```

### 2. Debian Package (.deb)
**Best for**: Debian/Ubuntu systems

```bash
./create_deb_package.sh
```

Creates:
- `dist/goeckoh-system_1.0.0_all.deb`

**Installation:**
```bash
sudo dpkg -i goeckoh-system_1.0.0_all.deb
sudo apt-get install -f  # Fix dependencies if needed
```

### 3. Standalone Bundle
**Best for**: Portable installations, no system-wide install

The package includes everything needed to run standalone.

## 📋 Package Contents

The deployment package includes:

### Core System
- ✅ Complete Python source code
- ✅ GOECKOH canonical system
- ✅ Rust library (libbio_audio.so)
- ✅ All Python modules and packages

### Assets
- ✅ STT models (Sherpa-ONNX)
- ✅ TTS models
- ✅ Voice profiles (if included)

### Launchers
- ✅ Desktop launcher script
- ✅ Desktop entry file (.desktop)
- ✅ Icon files

### Documentation
- ✅ README.md
- ✅ Installation instructions
- ✅ System documentation

### Scripts
- ✅ Installation script (install.sh)
- ✅ Launcher script (launch_bubble_system.sh)

## 🚀 Quick Deployment

### For End Users

1. **Download the package:**
   ```bash
   wget https://example.com/goeckoh-system-1.0.0.tar.gz
   ```

2. **Extract:**
   ```bash
   tar -xzf goeckoh-system-1.0.0.tar.gz
   cd goeckoh-system-1.0.0
   ```

3. **Install:**
   ```bash
   ./install.sh
   ```

4. **Launch:**
   - Search for "Goeckoh" in application menu
   - Or run: `goeckoh gui`

### For System Administrators

**DEB Package (Recommended for Ubuntu/Debian):**
```bash
sudo dpkg -i goeckoh-system_1.0.0_all.deb
sudo apt-get install -f
```

**Custom Location:**
```bash
sudo ./install.sh /opt/goeckoh
```

## 📦 Package Structure

```
goeckoh-system-1.0.0/
├── bin/
│   └── launch_bubble_system.sh
├── assets/
│   ├── model_stt/
│   └── model_tts/
├── GOECKOH/
│   └── goeckoh/
│       └── systems/
│           └── complete_unified_system.py
├── icons/
│   └── goeckoh-icon.png
├── install.sh
├── requirements.txt
├── README.md
└── [all Python source files]
```

## 🔧 Installation Locations

### Default Installation
- **Location**: `/opt/goeckoh`
- **Requires**: sudo/root
- **Command**: `sudo ./install.sh`

### User Installation
- **Location**: `~/goeckoh` (or custom)
- **Requires**: No sudo
- **Command**: `./install.sh ~/goeckoh`

### System-Wide (DEB)
- **Location**: `/opt/goeckoh`
- **Command**: `sudo dpkg -i goeckoh-system_1.0.0_all.deb`

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, or similar)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB free space
- **Audio**: Microphone and speakers

### Dependencies
The installer automatically installs:
- Python virtual environment
- All Python packages from requirements.txt
- Desktop launcher integration

### Optional Dependencies
- **Rust**: For building libbio_audio.so (if not included)
- **Node.js**: For Cognitive Nebula frontend (if building from source)
- **GPU**: For faster neural TTS (optional)

## 🎯 Post-Installation

After installation:

1. **Verify installation:**
   ```bash
   goeckoh help
   ```

2. **Test launch:**
   ```bash
   goeckoh gui
   ```

3. **Check desktop launcher:**
   - Search for "Goeckoh" in application menu
   - Icon should appear

## 🔍 Troubleshooting

### Installation Issues

**Missing dependencies:**
```bash
# For DEB package
sudo apt-get install -f

# For source package
sudo apt-get install python3 python3-pip python3-venv
```

**Permission errors:**
- Use `sudo` for system-wide installation
- Or install to user directory: `./install.sh ~/goeckoh`

**Icon not showing:**
```bash
update-desktop-database ~/.local/share/applications/
```

### Runtime Issues

**Python path issues:**
- Ensure virtual environment is activated
- Check PYTHONPATH is set correctly

**Audio issues:**
- Check audio devices: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- Install audio libraries: `sudo apt-get install portaudio19-dev`

## 📊 Package Sizes

- **Source package**: ~500MB-2GB (depending on assets)
- **DEB package**: ~500MB-2GB (compressed)
- **Installed size**: ~2-4GB (with dependencies)

## 🔐 Security Considerations

- Package includes only source code (no sensitive data)
- User data stored in `~/.goeckoh/` (created at runtime)
- No network access required for core functionality
- All dependencies from official repositories

## 📝 Distribution Checklist

Before distributing:

- [ ] Run `./package_deployment.sh` to create package
- [ ] Test installation on clean system
- [ ] Verify all dependencies are listed
- [ ] Check icon file is included
- [ ] Test desktop launcher
- [ ] Verify README is complete
- [ ] Check file permissions
- [ ] Test all launch modes

## 🚢 Release Process

1. **Version bump**: Update version in packaging scripts
2. **Package**: Run `./package_deployment.sh`
3. **Test**: Install and test on clean system
4. **Create DEB**: Run `./create_deb_package.sh` (optional)
5. **Upload**: Upload to distribution server
6. **Document**: Update release notes

---

**Ready to deploy!** Run `./package_deployment.sh` to create your deployment package.

