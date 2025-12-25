# Goeckoh Desktop Application

Goeckoh (pronounced "Go Echo") is an identity-matched speech replay system that helps neurodivergent individuals with speech support.

## What is Goeckoh?

Goeckoh listens to speech, gently corrects it, and replays it in the user's **own voice** — always in first person — so it feels like self-speech, not an outside command.

### Key Features

- **Identity-Matched Voice Replay**: Uses your own voice, not a synthetic one
- **Offline-First**: Works without internet connection for privacy and reliability
- **Real-Time Correction**: Gentle speech error correction and replay
- **Privacy-Forward**: All processing happens locally on your device

## Installation

### Prerequisites

- **Python 3.8 or higher** must be installed on your system
- Operating System: Windows, macOS, or Linux

### Windows

1. Download `Goeckoh-Setup-1.0.0.exe`
2. Run the installer
3. Follow the installation wizard
4. Launch Goeckoh from the Start Menu or Desktop shortcut

### macOS

1. Download `Goeckoh-1.0.0.dmg`
2. Open the DMG file
3. Drag Goeckoh to your Applications folder
4. Launch from Applications
5. If prompted about security, go to System Preferences → Security & Privacy and click "Open Anyway"

### Linux

1. Download `Goeckoh-1.0.0.AppImage` (or `.deb` for Debian/Ubuntu)
2. For AppImage:
   ```bash
   chmod +x Goeckoh-1.0.0.AppImage
   ./Goeckoh-1.0.0.AppImage
   ```
3. For DEB:
   ```bash
   sudo dpkg -i goeckoh-desktop_1.0.0_amd64.deb
   ```

## First Run

On first launch, Goeckoh will:

1. Check for required dependencies
2. Download necessary AI models (if not present)
3. Request microphone permissions
4. Start the speech processing backend

## Usage

1. **Launch the application**
2. **Grant microphone access** when prompted
3. **Start speaking** - Goeckoh will listen and process your speech
4. **Hear corrections** replayed in your own voice

## System Requirements

### Minimum

- **CPU**: Dual-core processor, 2 GHz or faster
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Microphone**: Required for speech input
- **Speakers/Headphones**: Required for audio output

### Recommended

- **CPU**: Quad-core processor, 2.5 GHz or faster
- **RAM**: 8 GB or more
- **Storage**: 5 GB free space (for models and voice profiles)
- **GPU**: Optional, but improves performance

## Troubleshooting

### "Python not found" error

Make sure Python 3.8 or higher is installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Install via Homebrew: `brew install python3`
- **Linux**: Install via package manager: `sudo apt install python3`

### Microphone not working

1. Check system permissions for microphone access
2. Verify your microphone is connected and working in other apps
3. Restart the application

### Models not downloading

1. Check your internet connection
2. Ensure you have sufficient disk space (at least 2 GB free)
3. Try manually running: `python desktop_app.py`

## Privacy

Goeckoh is designed with privacy as a core principle:

- **All processing happens locally** on your device
- **No internet connection required** for core functionality
- **No data is sent to external servers**
- **Voice profiles are stored locally** and encrypted

For more details, see our [Privacy Policy](../website/privacy.html).

## Support

- **Website**: https://goeckoh.com
- **Support**: See [support page](../website/support.html)
- **Issues**: Report on GitHub

## License

See LICENSE file for details.

## About the Name

"Goeckoh" is how "Go Echo" sounded when a neurodivergent individual said it — so the name itself starts with the voice we're protecting.
