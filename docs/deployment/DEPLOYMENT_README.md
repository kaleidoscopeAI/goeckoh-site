# 🚀 Goeckoh System - Deployment Package

## Quick Start

### Create Deployment Package

```bash
./package_deployment.sh
```

This creates a clean, distributable package in `dist/` directory.

### Create DEB Package (Ubuntu/Debian)

```bash
./create_deb_package.sh
```

Creates a `.deb` package for easy installation on Debian-based systems.

## 📦 What Gets Packaged

✅ **Core System**
- Complete Python source code
- GOECKOH canonical unified system
- Rust library (libbio_audio.so)
- All Python modules

✅ **Assets**
- STT models (Sherpa-ONNX)
- TTS models
- Voice profiles

✅ **Launchers**
- Desktop launcher script
- Desktop entry file
- Icon files

✅ **Documentation**
- README.md
- Installation guide
- System documentation

✅ **Installation Scripts**
- Automated installation
- Dependency management
- Desktop integration

## 📋 Package Files Created

After running `./package_deployment.sh`:

```
dist/
├── goeckoh-system-1.0.0.tar.gz    # Source package
└── goeckoh-system-1.0.0.zip      # Alternative format

build/
└── goeckoh-system-1.0.0/        # Package directory
    ├── install.sh               # Installation script
    ├── bin/
    │   └── launch_bubble_system.sh
    ├── assets/
    ├── GOECKOH/
    ├── icons/
    └── [all source files]
```

## 🎯 Installation Methods

### Method 1: Source Package

```bash
# Extract
tar -xzf goeckoh-system-1.0.0.tar.gz
cd goeckoh-system-1.0.0

# Install
./install.sh                    # To /opt/goeckoh (requires sudo)
./install.sh ~/goeckoh          # To user directory (no sudo)
```

### Method 2: DEB Package

```bash
sudo dpkg -i goeckoh-system_1.0.0_all.deb
sudo apt-get install -f  # Fix dependencies
```

### Method 3: Standalone (No Install)

```bash
# Extract and run directly
tar -xzf goeckoh-system-1.0.0.tar.gz
cd goeckoh-system-1.0.0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./bin/launch_bubble_system.sh gui
```

## 🔧 System Requirements

- **OS**: Linux (Ubuntu 20.04+, Debian 11+)
- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB free space
- **Audio**: Microphone and speakers

## 📝 Files Included

### Scripts
- `package_deployment.sh` - Create source package
- `create_deb_package.sh` - Create DEB package
- `install.sh` - Installation script (in package)
- `launch_bubble_system.sh` - Launcher script

### Documentation
- `DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `DEPLOYMENT_CHECKLIST.md` - Pre-deployment checklist
- `QUICK_START.md` - Quick start guide
- `SYSTEM_OVERVIEW.md` - System overview

## ✅ Pre-Packaging Checklist

Before creating the package:

- [x] Code consolidated
- [x] Desktop launcher created
- [x] Icon configured
- [x] All imports verified
- [ ] Icon file saved to `icons/goeckoh-icon.png` ⚠️ **DO THIS FIRST**
- [ ] Test launcher: `./launch_bubble_system.sh gui`

## 🚀 Deployment Steps

1. **Save your icon:**
   ```bash
   # Save your icon image to:
   cp /path/to/your/icon.png icons/goeckoh-icon.png
   ```

2. **Create package:**
   ```bash
   ./package_deployment.sh
   ```

3. **Test package:**
   ```bash
   cd build/goeckoh-system-1.0.0
   ./install.sh ~/test-goeckoh
   ```

4. **Create DEB (optional):**
   ```bash
   ./create_deb_package.sh
   ```

5. **Distribute:**
   - Upload `dist/goeckoh-system-1.0.0.tar.gz` to distribution server
   - Or share `dist/goeckoh-system_1.0.0_all.deb` for Debian systems

## 📊 Package Information

- **Version**: 1.0.0
- **Package Name**: goeckoh-system
- **Estimated Size**: 500MB - 2GB (depending on assets)
- **Installation Size**: 2-4GB (with dependencies)

## 🎯 Post-Installation

After installation, users can:

1. **Launch from menu**: Search for "Goeckoh"
2. **Launch from command**: `goeckoh gui`
3. **Access modes**: Right-click launcher for different modes

## 📖 Documentation

- `DEPLOYMENT_GUIDE.md` - Complete deployment documentation
- `SYSTEM_OVERVIEW.md` - What the system does
- `QUICK_START.md` - Quick start guide
- `INSTALL_LAUNCHER.md` - Desktop launcher setup

---

**Ready to package!** Run `./package_deployment.sh` to create your deployment package.

