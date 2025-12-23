# Cross-Platform Packaging - Quick Start

## 🚀 Package for All Platforms

### One Command (Interactive)
```bash
./package_all_platforms.sh
```

This will package for all available platforms with prompts.

### Individual Platforms

**Linux:**
```bash
./package_deployment.sh        # Source package
./create_deb_package.sh        # DEB package
```

**Windows:**
```bash
./package_windows.sh           # Portable ZIP + NSIS script
```

**macOS:**
```bash
./package_macos.sh             # .app bundle + DMG (requires macOS)
```

**Android:**
```bash
./package_android.sh           # APK (takes 10-30 min first time)
```

**iOS:**
```bash
./package_ios.sh               # Xcode project (requires macOS + Xcode)
```

## 📦 Output Locations

```
dist/
├── goeckoh-system-1.0.0.tar.gz              # Linux source
├── goeckoh-system_1.0.0_all.deb            # Linux DEB
├── windows/
│   └── goeckoh-system-1.0.0-windows-portable.zip
├── macos/
│   ├── goeckoh-system-1.0.0-macos.zip
│   └── goeckoh-system-1.0.0-macos.dmg
├── android/
│   └── goeckoh-1.0.0-debug.apk
└── ios/
    └── (Xcode project)
```

## ⚡ Quick Reference

| Platform | Script | Output | Time |
|----------|--------|--------|------|
| Linux | `./package_deployment.sh` | `.tar.gz` | ~1 min |
| Linux DEB | `./create_deb_package.sh` | `.deb` | ~2 min |
| Windows | `./package_windows.sh` | `.zip` | ~1 min |
| macOS | `./package_macos.sh` | `.dmg` | ~2 min |
| Android | `./package_android.sh` | `.apk` | 10-30 min |
| iOS | `./package_ios.sh` | Xcode project | ~5 min |

## 📋 Requirements by Platform

### All Platforms
- ✅ Icon saved to `icons/goeckoh-icon.png`

### Windows
- ✅ Can create on any system
- ⚠️ NSIS installer requires Windows or Wine

### macOS
- ⚠️ Requires macOS for DMG creation
- ✅ Can create ZIP on any system

### Android
- ✅ Works on Linux/Mac/Windows
- ✅ Requires: `pip install buildozer`
- ⏱️ First build downloads SDK (10-30 min)

### iOS
- ⚠️ Requires macOS with Xcode
- ✅ Requires: `pip install briefcase` or `kivy-ios`
- ✅ Requires Apple Developer account (for distribution)

## 🎯 Installation Methods

### Linux
```bash
# DEB
sudo dpkg -i goeckoh-system_1.0.0_all.deb

# Source
tar -xzf goeckoh-system-1.0.0.tar.gz
cd goeckoh-system-1.0.0
./install.sh
```

### Windows
- **Portable**: Extract ZIP, run `goeckoh.bat`
- **Installer**: Run `.exe`, follow wizard

### macOS
- Drag `.app` to Applications folder
- Or open DMG and drag to Applications

### Android
```bash
adb install goeckoh-1.0.0-debug.apk
# Or transfer APK to device and install
```

### iOS
- Open Xcode project
- Build and archive
- Distribute via App Store or TestFlight

---

**See `CROSS_PLATFORM_PACKAGING.md` for complete details!**

