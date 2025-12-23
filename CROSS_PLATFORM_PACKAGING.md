# Cross-Platform Packaging Guide

## 🎯 Supported Platforms

The Goeckoh System can be packaged for:

- ✅ **Linux** (Ubuntu/Debian) - `.deb` and source packages
- ✅ **Windows** - Portable ZIP and NSIS installer
- ✅ **macOS** - `.app` bundle and `.dmg` installer
- ✅ **Android** - APK via Buildozer
- ✅ **iOS** - IPA via Briefcase or Kivy iOS

## 📦 Platform-Specific Packages

### Linux

**Source Package:**
```bash
./package_deployment.sh
```

**DEB Package:**
```bash
./create_deb_package.sh
```

**Output:**
- `dist/goeckoh-system-1.0.0.tar.gz`
- `dist/goeckoh-system_1.0.0_all.deb`

### Windows

**Create Package:**
```bash
./package_windows.sh
```

**Output:**
- `dist/windows/goeckoh-system-1.0.0-windows-portable.zip`
- `build/windows/goeckoh-installer.nsi` (NSIS installer script)

**To Create Installer:**
1. Install NSIS: https://nsis.sourceforge.io/
2. Compile: `makensis build/windows/goeckoh-installer.nsi`

**Installation:**
- **Portable**: Extract ZIP, run `goeckoh.bat`
- **Installer**: Run `.exe` installer, follow wizard

### macOS

**Create Package:**
```bash
./package_macos.sh
```

**Output:**
- `dist/macos/goeckoh-system-1.0.0-macos.zip`
- `dist/macos/goeckoh-system-1.0.0-macos.dmg` (if on macOS)

**Installation:**
1. Open DMG file
2. Drag app to Applications folder
3. Launch from Applications

**Requirements:**
- macOS 10.13 (High Sierra) or later
- Python 3.8+ (install via Homebrew: `brew install python3`)

### Android

**Create APK:**
```bash
./package_android.sh
```

**Debug APK:**
```bash
./package_android.sh debug
```

**Release APK (signed):**
```bash
./package_android.sh release
```

**Output:**
- `dist/android/goeckoh-1.0.0-debug.apk`
- `dist/android/goeckoh-1.0.0-release.apk`

**Installation:**
```bash
# Via ADB
adb install dist/android/goeckoh-1.0.0-debug.apk

# Or transfer APK to device and install manually
```

**Requirements:**
- Buildozer: `pip install buildozer`
- Android SDK/NDK (auto-downloaded by Buildozer)
- Java JDK

**First Build:**
- Takes 10-30 minutes (downloads SDK/NDK)
- Requires internet connection
- Subsequent builds are faster

### iOS

**Create Package:**
```bash
./package_ios.sh
```

**Requirements:**
- macOS with Xcode installed
- Apple Developer account (for distribution)
- BeeWare Briefcase: `pip install briefcase`
- Or Kivy iOS: `pip install kivy-ios`

**Build Process:**
1. Run `./package_ios.sh`
2. Open Xcode project
3. Configure code signing
4. Build and archive
5. Export IPA

**Output:**
- Xcode project in `build/ios/`
- IPA file after Xcode build

## 🔧 Platform-Specific Considerations

### Windows
- Uses `.bat` launcher scripts
- Requires Python 3.8+ installed separately
- NSIS installer creates Start Menu shortcuts
- Portable version requires no installation

### macOS
- Creates `.app` bundle (native macOS format)
- Requires Python 3.8+ (via Homebrew recommended)
- DMG provides drag-and-drop installation
- Requests microphone permissions on first launch

### Android
- Uses Kivy framework (mobile-optimized)
- Buildozer handles all Android dependencies
- APK can be sideloaded or distributed via Play Store
- Requires Android 5.0 (API 21) or later

### iOS
- Requires macOS and Xcode
- Needs Apple Developer account for distribution
- App Store submission requires additional steps
- Supports iOS 11.0 or later

## 📋 Packaging Checklist

### All Platforms
- [ ] Save icon to `icons/goeckoh-icon.png`
- [ ] Test launcher on target platform
- [ ] Verify all dependencies listed
- [ ] Check file permissions

### Windows Specific
- [ ] Test `.bat` launcher
- [ ] Verify Python detection
- [ ] Test NSIS installer (if creating)

### macOS Specific
- [ ] Test `.app` bundle
- [ ] Verify icon displays
- [ ] Test DMG creation
- [ ] Check code signing (for distribution)

### Android Specific
- [ ] Install Buildozer
- [ ] Test APK on device
- [ ] Verify permissions in manifest
- [ ] Test on multiple Android versions

### iOS Specific
- [ ] Install Briefcase or Kivy iOS
- [ ] Configure Xcode project
- [ ] Set up code signing
- [ ] Test on iOS device/simulator

## 🚀 Quick Reference

### Create All Packages

```bash
# Linux
./package_deployment.sh
./create_deb_package.sh

# Windows (on Linux/Mac, creates portable package)
./package_windows.sh

# macOS (on macOS)
./package_macos.sh

# Android
./package_android.sh

# iOS (on macOS with Xcode)
./package_ios.sh
```

### Package Locations

```
dist/
├── goeckoh-system-1.0.0.tar.gz          # Linux source
├── goeckoh-system_1.0.0_all.deb        # Linux DEB
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

## 📱 Mobile-Specific Features

### Android
- Touch-optimized UI (Kivy)
- Mobile audio handling
- Background processing
- Notification support
- File system access

### iOS
- Native iOS UI elements
- Core Audio integration
- Background audio
- App Store compliance
- iCloud integration (optional)

## 🔐 Code Signing

### macOS
- Required for distribution outside App Store
- Use Xcode: Product > Archive > Distribute App
- Or command line: `codesign` and `productbuild`

### iOS
- Required for all iOS apps
- Configure in Xcode: Signing & Capabilities
- Requires Apple Developer account ($99/year)

### Android
- Required for Play Store
- Use `jarsigner` or `apksigner`
- Debug builds are self-signed

## 📊 Platform Comparison

| Feature | Linux | Windows | macOS | Android | iOS |
|---------|-------|---------|-------|---------|-----|
| Package Format | .deb, .tar.gz | .exe, .zip | .dmg, .app | .apk | .ipa |
| Python Included | No | No | No | Yes (via Buildozer) | Yes (via Briefcase) |
| Installation | Easy | Easy | Drag & Drop | APK install | App Store/Xcode |
| Code Signing | Optional | Optional | Recommended | Required (Play Store) | Required |
| Build Time | Fast | Fast | Fast | Slow (first time) | Slow |

## 🎯 Distribution Recommendations

- **Linux**: DEB package for Debian/Ubuntu, tar.gz for others
- **Windows**: NSIS installer for most users, portable ZIP for advanced
- **macOS**: DMG for easy distribution, ZIP as backup
- **Android**: APK for direct install, Play Store for wide distribution
- **iOS**: App Store for public, TestFlight for beta, Enterprise for internal

---

**Ready to package for all platforms!** Run the appropriate script for each target platform.

