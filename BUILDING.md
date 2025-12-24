# Building Goeckoh Desktop Application

This document explains how to build distributable packages of the Goeckoh desktop application for Windows, macOS, and Linux.

## Prerequisites

Before building, ensure you have:

1. **Node.js** (v16 or higher) and **npm** installed
2. **Python 3.8+** installed and accessible in PATH
3. **Git** (for version control)

### Platform-Specific Requirements

#### Windows
- Windows 10 or later
- No additional requirements for building Windows packages on Windows

#### macOS
- macOS 10.13 or later
- Xcode Command Line Tools: `xcode-select --install`
- For code signing: Apple Developer account and certificates

#### Linux
- Ubuntu 18.04+ / Debian 10+ / Fedora 30+ or equivalent
- For building DEB packages: `dpkg` and `fakeroot`
- For building RPM packages: `rpm-build`

## Quick Start

### Build for Current Platform

```bash
# Navigate to project root
cd /path/to/goeckoh-site

# Run the build script
./build_desktop_app.sh
```

### Build for Specific Platform

```bash
# Windows
./build_desktop_app.sh windows

# macOS
./build_desktop_app.sh macos

# Linux
./build_desktop_app.sh linux

# All platforms
./build_desktop_app.sh all
```

## Manual Build Process

If you prefer to build manually:

```bash
# 1. Navigate to electron-app directory
cd electron-app

# 2. Install dependencies
npm install

# 3. Build for your platform
npm run dist          # Current platform
npm run dist:win      # Windows
npm run dist:mac      # macOS
npm run dist:linux    # Linux
```

## Build Output

Build artifacts are created in `dist/electron/`:

```
dist/electron/
├── Goeckoh-Setup-1.0.0.exe          # Windows installer
├── Goeckoh-1.0.0.dmg                # macOS disk image
├── Goeckoh-1.0.0.AppImage           # Linux AppImage
├── goeckoh-desktop_1.0.0_amd64.deb # Debian/Ubuntu package
└── unpacked/                        # Unpacked application files
```

## Electron Builder Configuration

The build configuration is in `electron-app/package.json` under the `build` section:

### Windows Configuration

```json
{
  "win": {
    "target": ["nsis"],
    "icon": "build/icon.ico"
  },
  "nsis": {
    "oneClick": false,
    "allowToChangeInstallationDirectory": true
  }
}
```

Creates a standard Windows installer with options for installation directory.

### macOS Configuration

```json
{
  "mac": {
    "target": ["dmg"],
    "icon": "build/icon.icns",
    "category": "public.app-category.healthcare-fitness"
  }
}
```

Creates a DMG disk image for drag-and-drop installation.

### Linux Configuration

```json
{
  "linux": {
    "target": ["AppImage", "deb"],
    "icon": "build/icon.png",
    "category": "Utility"
  }
}
```

Creates both AppImage (portable) and DEB (installable) packages.

## Code Signing

### macOS Code Signing

For distribution outside the Mac App Store:

1. Get Apple Developer certificates
2. Set environment variables:
   ```bash
   export CSC_LINK=/path/to/certificate.p12
   export CSC_KEY_PASSWORD=your_password
   ```
3. Build: `npm run dist:mac`

### Windows Code Signing

For production releases:

1. Obtain a code signing certificate
2. Set environment variables:
   ```bash
   export CSC_LINK=/path/to/certificate.pfx
   export CSC_KEY_PASSWORD=your_password
   ```
3. Build: `npm run dist:win`

## Application Structure

The packaged application includes:

```
Goeckoh/
├── electron-app/
│   ├── main.js              # Electron main process
│   ├── index.html           # UI interface
│   └── package.json         # Electron configuration
├── desktop_app.py           # Python entry point
├── main_app.py              # Application logic
├── src/                     # Python modules
│   ├── neuro_backend.py     # Speech processing
│   ├── config.py            # Configuration
│   └── ...
├── assets/                  # AI models
│   ├── model_stt/           # Speech-to-text models
│   └── model_tts/           # Text-to-speech models
└── requirements.txt         # Python dependencies
```

## Testing the Build

### Before Distribution

1. **Test on clean system**: Install on a system without development tools
2. **Verify Python detection**: Check that bundled Python or system Python is found
3. **Test microphone access**: Ensure permissions are requested correctly
4. **Check model loading**: Verify AI models are loaded properly
5. **Test speech pipeline**: Complete end-to-end speech processing test

### Automated Tests

```bash
# Run Python tests
python -m pytest tests/

# Validate package integrity
cd dist/electron
# Windows: Run installer in silent mode
# macOS: Mount DMG and verify contents
# Linux: Install DEB and verify files
```

## Distribution

### GitHub Releases

1. Create a new release on GitHub
2. Upload build artifacts:
   - `Goeckoh-Setup-1.0.0.exe` (Windows)
   - `Goeckoh-1.0.0.dmg` (macOS)
   - `Goeckoh-1.0.0.AppImage` (Linux)
   - `goeckoh-desktop_1.0.0_amd64.deb` (Debian/Ubuntu)

### Website Download Page

Update `website/download.html` with download links:

```html
<a href="https://github.com/kaleidoscopeAI/goeckoh-site/releases/download/v1.0.0/Goeckoh-Setup-1.0.0.exe">
  Download for Windows
</a>
```

## Troubleshooting

### Build Fails on macOS

**Issue**: "Cannot sign application"
**Solution**: Either skip code signing for testing or set up proper certificates

```bash
# Build without code signing (for testing only)
export CSC_IDENTITY_AUTO_DISCOVERY=false
npm run dist:mac
```

### Build Size Too Large

**Issue**: Package is several GB
**Solution**: Exclude unnecessary files in `package.json`:

```json
{
  "build": {
    "files": [
      "!**/node_modules/**/*",
      "!assets/unused/**"
    ]
  }
}
```

### Python Not Found at Runtime

**Issue**: Application can't find Python
**Solution**: Update `main.js` to include more Python search paths or bundle Python with the app

## Advanced Configuration

### Bundling Python

To include Python runtime with the application:

1. Download Python embeddable package
2. Place in `electron-app/python/`
3. Update `main.js` to use bundled Python
4. Add to `extraResources` in `package.json`

### Custom Installers

Modify NSIS script for Windows:
- Edit `electron-app/build/installer.nsi`
- Add custom pages, shortcuts, registry keys

## Support

For build issues:
- Check Electron Builder docs: https://www.electron.build/
- Open an issue on GitHub
- Contact support team

## License

See LICENSE file for build and distribution terms.
