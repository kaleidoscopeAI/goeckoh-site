# Goeckoh Desktop Application - Quick Start

Welcome to Goeckoh (pronounced "Go Echo"), your identity-matched speech replay system!

## What You Downloaded

You've downloaded the Goeckoh desktop application, which helps with speech support by:
- Listening to your speech
- Gently correcting it
- Replaying it in **your own voice**

## Before You Start

### Required: Python 3.8 or Higher

Goeckoh requires Python to be installed on your system. Don't have it? Here's how to install:

#### Windows
1. Download from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify: Open Command Prompt and type `python --version`

#### macOS
```bash
# Install Homebrew first (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3

# Verify
python3 --version
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# Fedora
sudo dnf install python3 python3-pip

# Verify
python3 --version
```

## Installation

### Windows

1. **Run the installer**: Double-click `Goeckoh-Setup-1.0.0.exe`
2. **Follow the wizard**: Choose installation location
3. **Launch**: Find Goeckoh in Start Menu or Desktop shortcut

### macOS

1. **Open the DMG**: Double-click `Goeckoh-1.0.0.dmg`
2. **Drag to Applications**: Drag Goeckoh to the Applications folder
3. **First launch**: Right-click Goeckoh → Open (to bypass Gatekeeper)
4. **Grant permissions**: Allow microphone access when prompted

### Linux (AppImage)

```bash
# Make executable
chmod +x Goeckoh-1.0.0.AppImage

# Run
./Goeckoh-1.0.0.AppImage
```

### Linux (DEB Package)

```bash
# Install
sudo dpkg -i goeckoh-desktop_1.0.0_amd64.deb

# If dependencies are missing
sudo apt-get install -f

# Run from applications menu or
goeckoh
```

## First Run

When you launch Goeckoh for the first time:

1. **Microphone Permission**: Grant access when prompted
2. **Model Download**: AI models will download automatically (requires internet)
3. **Setup Complete**: Wait for "System Running" status

This may take a few minutes on first launch.

## Using Goeckoh

### Basic Operation

1. **Launch the app**: Open Goeckoh from your applications
2. **Wait for "System Running"**: Status indicator shows green
3. **Start speaking**: Just talk naturally
4. **Listen**: Hear your corrected speech in your own voice

### Tips for Best Results

- **Speak clearly**: But don't worry about being perfect
- **Use a good microphone**: Built-in mics work, but external is better
- **Quiet environment**: Reduces background noise
- **Be patient**: The system learns your voice over time

## Troubleshooting

### "Python not found"

**Problem**: Application can't find Python
**Solution**: 
1. Make sure Python 3.8+ is installed
2. Add Python to your system PATH
3. Restart the application

### Microphone not working

**Problem**: No audio input detected
**Solution**:
1. Check system microphone permissions
2. Test microphone in other apps
3. Select correct input device in system settings
4. Restart Goeckoh

### "Failed to download models"

**Problem**: AI models won't download
**Solution**:
1. Check internet connection
2. Ensure you have 2GB free disk space
3. Temporarily disable firewall/antivirus
4. Manual download: See [Manual Setup](#manual-model-download)

### Application won't start

**Problem**: Goeckoh crashes on launch
**Solution**:
1. Check Python version: `python3 --version`
2. Run from terminal to see error messages
3. Check system requirements
4. Reinstall the application

## Manual Model Download

If automatic download fails, you can manually download models:

1. Download models from: [GitHub Releases](https://github.com/kaleidoscopeAI/goeckoh-site/releases)
2. Extract to application folder:
   - **Windows**: `C:\Program Files\Goeckoh\resources\assets\`
   - **macOS**: `/Applications/Goeckoh.app/Contents/Resources/assets/`
   - **Linux**: `~/.local/share/goeckoh/assets/`

## System Requirements

### Minimum
- **OS**: Windows 10, macOS 10.13, Ubuntu 18.04 (or equivalent)
- **CPU**: 2+ cores, 2 GHz
- **RAM**: 4 GB
- **Storage**: 2 GB free
- **Microphone**: Required
- **Internet**: For initial model download only

### Recommended
- **CPU**: 4+ cores, 2.5 GHz or faster
- **RAM**: 8 GB or more
- **Storage**: 5 GB free
- **GPU**: Optional (improves performance)

## Getting Help

### Documentation
- **Full README**: See `electron-app/README.md` in installation folder
- **Website**: https://goeckoh.com
- **Support Page**: https://goeckoh.com/support.html

### Community & Support
- **GitHub Issues**: Report bugs and request features
- **Email Support**: See support page for contact info

## Privacy & Security

Goeckoh is designed with your privacy in mind:

- ✅ **All processing happens locally** on your device
- ✅ **No internet required** after initial setup
- ✅ **No data sent to external servers**
- ✅ **Voice profiles stored locally** and encrypted
- ✅ **Open source** - you can verify the code

For full details, see our [Privacy Policy](https://goeckoh.com/privacy.html).

## Updates

Goeckoh will check for updates automatically (requires internet):
- **Auto-update**: Enabled by default
- **Manual check**: Help → Check for Updates
- **Release notes**: See GitHub releases

## Uninstalling

### Windows
1. Settings → Apps → Goeckoh → Uninstall
2. Or use Control Panel → Programs and Features

### macOS
1. Drag Goeckoh from Applications to Trash
2. Remove data: `rm -rf ~/Library/Application Support/goeckoh`

### Linux
```bash
# DEB package
sudo apt remove goeckoh-desktop

# AppImage
# Simply delete the AppImage file
```

## Next Steps

1. **Complete first run setup**
2. **Test with simple phrases**
3. **Explore the interface**
4. **Read the full documentation**
5. **Provide feedback** (helps us improve!)

## About

Goeckoh (pronounced "Go Echo") - The name itself comes from how a neurodivergent individual pronounced "Go Echo", honoring the voices we're protecting.

Built with ❤️ for the neurodivergent community.

---

**Welcome to Goeckoh!** We're excited to support your communication journey.
