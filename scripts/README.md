# Scripts Directory

This directory contains build, deployment, and utility scripts for the Goeckoh system.

## Build Scripts

### Desktop Applications
- **`build_desktop.sh`** - Build desktop application
- **`create_deb_package.sh`** - Create Debian package for Linux

### Cross-Platform Packaging
- **`package_all_platforms.sh`** - Package for all supported platforms
- **`package_android.sh`** - Android mobile app packaging
- **`package_ios.sh`** - iOS mobile app packaging
- **`package_macos.sh`** - macOS desktop packaging
- **`package_windows.sh`** - Windows desktop packaging
- **`package_deployment.sh`** - Production deployment packaging

## System Management

### Launcher Scripts
- **`run_system.sh`** - Run the system locally (development)
- **`launch_bubble_system.sh`** - Launch the bubble system interface

### Installation and Setup
- **`install_launcher.sh`** - Install system launcher
- **`setup_icon.sh`** - Set up application icons

### Deployment
- **`deploy_system.sh`** - Automated deployment script
- **`verify_merge.sh`** - Verify repository merge integrity

## Utility Scripts

### Python Utilities
- **`env_check.py`** - Check Python environment and dependencies
- **`extract_code_from_txt.py`** - Extract code from text files
- **`attempt_analysis.py`** - Analysis utility
- **`deepseek_python_20251219_8ca7f9.py`** - DeepSeek integration script

## Usage

### Running Scripts

Most shell scripts should be run from the repository root:

```bash
# From repository root
./scripts/run_system.sh

# Or make them executable first
chmod +x scripts/*.sh
./scripts/build_desktop.sh
```

### Python Utilities

Python scripts can be run directly:

```bash
# Check environment
python scripts/env_check.py

# Extract code from files
python scripts/extract_code_from_txt.py <input_file>
```

### Build and Package

```bash
# Build for current platform
./scripts/build_desktop.sh

# Package for all platforms
./scripts/package_all_platforms.sh

# Package for specific platform
./scripts/package_android.sh
./scripts/package_macos.sh
./scripts/package_windows.sh
```

### Deployment

```bash
# Deploy to production
./scripts/deploy_system.sh

# Verify deployment
./scripts/verify_merge.sh
```

## Script Maintenance

When modifying scripts:

1. **Keep scripts idempotent** - Safe to run multiple times
2. **Add error handling** - Check for prerequisites and fail gracefully
3. **Document assumptions** - Note required environment variables or tools
4. **Test on target platforms** - Verify platform-specific scripts work
5. **Update this README** - Document new scripts or significant changes

## Platform Requirements

### All Platforms
- Bash (for shell scripts)
- Python 3.8+ (for Python scripts)
- Git

### Platform-Specific
- **Android**: Android SDK, Buildozer
- **iOS**: Xcode, iOS development tools (macOS only)
- **macOS**: Xcode command line tools
- **Windows**: WSL or Git Bash for shell scripts
- **Linux**: Standard build tools (make, gcc, etc.)

## See Also

- Deployment Guide: [../docs/deployment/DEPLOYMENT_GUIDE.md](../docs/deployment/DEPLOYMENT_GUIDE.md)
- Cross-Platform Packaging: [../docs/deployment/CROSS_PLATFORM_PACKAGING.md](../docs/deployment/CROSS_PLATFORM_PACKAGING.md)
- Main README: [../README.md](../README.md)
