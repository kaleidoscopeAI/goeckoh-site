# GOECKOH (Go Echo)

**Identity-Matched Speech Replay System for Neurodivergent Individuals**

This repository contains both the Goeckoh application code and its marketing website.

## Repository Structure

```
.
├── website/               # Marketing and information website
│   ├── index.html        # Main landing page
│   ├── download.html     # Download page
│   ├── privacy.html      # Privacy policy
│   ├── terms.html        # Terms of service
│   ├── support.html      # Support information
│   └── images/           # Website images and assets
│
├── docs/                 # Documentation (see docs/INDEX.md)
│   ├── deployment/       # Deployment and packaging guides
│   ├── system/          # System architecture and status
│   └── guides/          # User guides and setup instructions
│
├── config/              # Configuration files
│   ├── config.yaml      # Main configuration
│   ├── config.schema.yaml  # Configuration schema
│   └── ...              # Other config files
│
├── scripts/             # Build and deployment scripts
│
├── cognitive-nebula/    # 3D visualization frontend (React + Three.js)
│
├── GOECKOH/             # Main application package
│   ├── goeckoh/         # Core Python application
│   ├── frontend/        # React frontend
│   └── rust_core/       # Rust performance core
│
├── mobile/              # Mobile platform code
│   ├── ios/            # iOS Swift code
│   └── android/        # Android Kotlin code
│
├── assets/              # Application assets (models, icons, etc.)
│   ├── model_stt/      # Speech-to-text models
│   └── model_tts/      # Text-to-speech models
│
├── tests/               # Test files
├── archive/             # Archived/reference materials
│   ├── research/       # Research papers and chat logs
│   ├── old-docs/       # Old documentation
│   └── media/          # Archive media files
│
└── [core Python files] # Core application Python files at root
```

## About Goeckoh

Goeckoh (pronounced "Go Echo") is an offline-first speech support system that gently corrects speech and replays it in the user's own voice, in first person. It's designed specifically for neurodivergent individuals who need speech assistance.

### Key Features

- **Identity-Matched Voice Replay**: Uses your own voice, not a synthetic one
- **Offline-First**: Works without internet connection for privacy and reliability
- **Privacy-Forward**: All processing happens locally on your device
- **Real-Time Correction**: Gentle speech error correction and replay

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for frontend development)
- Git

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/kaleidoscopeAI/goeckoh-site.git
   cd goeckoh-site
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure the system:
   ```bash
   # Configuration files are in the config/ directory
   # Main configuration: config/config.yaml
   # Validate configuration:
   python -m cli validate
   ```

4. Run the application:
   ```bash
   python -m cli start
   ```

### CLI Commands

- `python -m cli validate` - Validate config.yaml against config.schema.yaml
- `python -m cli fix` - Auto-fix common issues in config.yaml
- `python -m cli start` - Start a REPL with the agent

### Voice Cloning (Required for Speak Command)

The CLI enforces voice cloning with no fallback to local TTS. You MUST provide a clean WAV sample of your voice:

- Via config: Set `voice_profile_path` in `config/config.yaml`
- Via CLI: Use `--voice-profile` flag

**Requirements:**
- Minimum duration: 5 seconds
- Format: 16kHz WAV preferred

**Examples:**
```bash
# Record 5s sample and use for cloning
python -m cli speak --record --duration 5 --voice-profile ./sample_voice.wav

# Use existing WAV sample
python -m cli speak --input-file ./input.wav --voice-profile ./sample_voice.wav
```

## Configuration

The system uses YAML-based configuration files located in the `config/` directory:

- **`config/config.yaml`** - Main system configuration
- **`config/config.schema.yaml`** - Configuration schema for validation
- **`config/guardian_policy.json`** - Safety and guardian policies
- **`config/buildozer.spec`** - Mobile build configuration

### Configuration Management

```bash
# Validate your configuration
python -m cli validate

# Auto-fix common configuration issues
python -m cli fix
```

See `config/config.yaml` for all available configuration options.

### Read Documents

```bash
# Read all .txt/.md/.pdf files from a folder recursively
python -m cli read-docs --path ./documents --recursive

# Non-recursive (top-level only)
python -m cli read-docs --path ./documents --no-recursive
```

### Read Code

```bash
# Read all .py files from a folder recursively
python -m cli read-code --path ./project --recursive
```

## Website Development

The marketing website is located in the `website/` directory. To view it locally:

1. Open `website/index.html` in your browser, or
2. Use a local server:
   ```bash
   cd website
   python -m http.server 8000
   # Visit http://localhost:8000
   ```

## Documentation

For comprehensive documentation, see the [`docs/`](docs/) directory:

- **[Documentation Index](docs/INDEX.md)** - Complete documentation overview
- **[System Overview](docs/system/SYSTEM_OVERVIEW.md)** - System architecture and features
- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Getting started quickly
- **[Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Cross-Platform Packaging](docs/deployment/CROSS_PLATFORM_PACKAGING.md)** - Building for multiple platforms

### Build and Deployment Scripts

All build and deployment scripts are located in the `scripts/` directory:
- `scripts/run_system.sh` - Run the system locally
- `scripts/build_desktop.sh` - Build desktop application
- `scripts/package_*.sh` - Platform-specific packaging scripts
- `scripts/deploy_system.sh` - Deployment automation

## Technical Stack

- **Backend**: Python with voice cloning (Coqui TTS, resemblyzer)
- **Frontend**: React, TypeScript
- **Voice Processing**: Real-time audio processing and voice cloning
- **Platform Support**: Linux, macOS, Windows, iOS, Android

## Notes

- Voice cloning uses **Coqui TTS** and **resemblyzer** - these are large packages and may require a GPU for fast synthesis
- If voice profile is missing or libraries cannot be initialized, the CLI will exit with an error
- For accurate cloning, provide a clean voice sample (≥5s) in 16kHz WAV format

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Repository organization guidelines
- Development setup instructions
- Code and documentation standards
- Pull request process

Please feel free to submit issues and pull requests.

## License

[License information to be added]

## Support

- Visit our [support page](website/support.html)
- Check the [documentation](docs/)
- File an issue on GitHub

## Privacy

We take privacy seriously. See our [Privacy Policy](website/privacy.html) for details.

## Links

- Website: https://goeckoh.com/
- Repository: https://github.com/kaleidoscopeAI/goeckoh-site
