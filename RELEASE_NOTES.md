# Release Notes - Goeckoh v1.0.0

## 🎉 First Release - Desktop Application

We're excited to announce the first public release of Goeckoh Desktop, an identity-matched speech replay system designed for neurodivergent individuals.

### What's New

#### Core Features
- ✅ **Identity-Matched Voice Replay**: Speech correction replayed in your own voice
- ✅ **Real-time Processing**: Low-latency speech recognition and synthesis
- ✅ **Offline-First**: All processing happens locally for privacy
- ✅ **Cross-Platform**: Available for Windows, macOS, and Linux

#### Desktop Application
- 🖥️ Native desktop app using Electron
- 🎨 Clean, accessible user interface
- 📊 Real-time status indicators
- 🔊 Audio processing pipeline visualization

#### System Components
- 🧠 Neural speech recognition (Sherpa-ONNX)
- 🗣️ Text-to-speech synthesis (Piper)
- ✏️ Grammar correction engine
- 💎 Crystalline Heart processing core

### Downloads

Choose your platform:

#### Windows
- **Installer**: [Goeckoh-Setup-1.0.0.exe](https://github.com/kaleidoscopeAI/goeckoh-site/releases/download/v1.0.0/Goeckoh-Setup-1.0.0.exe)
- **Size**: ~150 MB
- **Requirements**: Windows 10+, Python 3.8+

#### macOS
- **DMG**: [Goeckoh-1.0.0.dmg](https://github.com/kaleidoscopeAI/goeckoh-site/releases/download/v1.0.0/Goeckoh-1.0.0.dmg)
- **Size**: ~150 MB
- **Requirements**: macOS 10.13+, Python 3.8+
- **Note**: Not code-signed yet; right-click → Open to bypass Gatekeeper

#### Linux
- **AppImage**: [Goeckoh-1.0.0.AppImage](https://github.com/kaleidoscopeAI/goeckoh-site/releases/download/v1.0.0/Goeckoh-1.0.0.AppImage)
- **DEB Package**: [goeckoh-desktop_1.0.0_amd64.deb](https://github.com/kaleidoscopeAI/goeckoh-site/releases/download/v1.0.0/goeckoh-desktop_1.0.0_amd64.deb)
- **Size**: ~150 MB
- **Requirements**: Ubuntu 18.04+, Python 3.8+

### Installation

See [QUICKSTART.md](QUICKSTART.md) for detailed installation instructions.

**Quick Install:**

```bash
# Windows: Run the installer
Goeckoh-Setup-1.0.0.exe

# macOS: Open DMG and drag to Applications
open Goeckoh-1.0.0.dmg

# Linux (AppImage)
chmod +x Goeckoh-1.0.0.AppImage
./Goeckoh-1.0.0.AppImage

# Linux (DEB)
sudo dpkg -i goeckoh-desktop_1.0.0_amd64.deb
```

### System Requirements

**Minimum:**
- CPU: Dual-core, 2 GHz
- RAM: 4 GB
- Storage: 2 GB free
- Microphone required

**Recommended:**
- CPU: Quad-core, 2.5 GHz or faster
- RAM: 8 GB
- Storage: 5 GB free
- GPU (optional, improves performance)

### Known Issues

1. **First Launch Delay**: Initial model download may take 5-10 minutes
2. **macOS Gatekeeper**: Not code-signed; requires right-click → Open
3. **Python Dependency**: Must have Python 3.8+ installed separately
4. **Voice Profile**: First-time voice training requires ~5 minutes of speech samples

### Roadmap

#### Coming in v1.1.0
- [ ] Bundled Python runtime (no separate installation needed)
- [ ] Code signing for macOS and Windows
- [ ] Improved voice cloning accuracy
- [ ] Multi-language support
- [ ] Voice profile management UI

#### Future Versions
- [ ] Mobile apps (iOS, Android)
- [ ] Web interface option
- [ ] Advanced voice customization
- [ ] Clinical dashboard for therapists
- [ ] Usage analytics and progress tracking

### Building from Source

Want to build it yourself?

```bash
# Clone the repository
git clone https://github.com/kaleidoscopeAI/goeckoh-site.git
cd goeckoh-site

# Install dependencies
pip install -r requirements.txt
cd electron-app
npm install

# Build for your platform
npm run dist
```

See [BUILDING.md](BUILDING.md) for complete build instructions.

### Contributing

We welcome contributions! Areas where you can help:

- 🐛 Bug reports and fixes
- 📝 Documentation improvements
- 🌍 Translations
- 🎨 UI/UX enhancements
- 🧪 Testing on different platforms

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Privacy & Security

Goeckoh is built with privacy as a core principle:

- All speech processing happens **locally on your device**
- No internet connection required after initial setup
- No data sent to external servers
- Voice profiles are encrypted and stored locally
- Open source - verify the code yourself

Read our full [Privacy Policy](https://goeckoh.com/privacy.html).

### Support

Need help?

- 📚 **Documentation**: [README.md](README.md), [QUICKSTART.md](QUICKSTART.md)
- 💬 **Community**: [GitHub Discussions](https://github.com/kaleidoscopeAI/goeckoh-site/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/kaleidoscopeAI/goeckoh-site/issues)
- 📧 **Email**: See [support page](https://goeckoh.com/support.html)

### Acknowledgments

Special thanks to:

- The neurodivergent community for feedback and inspiration
- Contributors to Sherpa-ONNX and Piper TTS
- Everyone who tested early versions

### License

[License details to be added]

---

**About the Name**: "Goeckoh" is how "Go Echo" sounded when a neurodivergent individual said it — so the name itself starts with the voice we're protecting.

Built with ❤️ for accessibility and inclusion.

### Changelog

**v1.0.0** (2024-12-24)
- Initial public release
- Electron desktop application
- Cross-platform support (Windows, macOS, Linux)
- Core speech processing pipeline
- Offline-first architecture
- Identity-matched voice replay
