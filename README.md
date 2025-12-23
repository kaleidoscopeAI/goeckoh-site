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
├── src/                  # Application source code
├── docs/                 # Documentation
├── tests/                # Test files
├── assets/               # Application assets (models, etc.)
└── ...                   # Other application files
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

3. Run the application:
   ```bash
   python -m cli start
   ```

### CLI Commands

- `python -m cli validate` - Validate config.yaml against config.schema.yaml
- `python -m cli fix` - Auto-fix common issues in config.yaml
- `python -m cli start` - Start a REPL with the agent

### Voice Cloning (Required for Speak Command)

The CLI enforces voice cloning with no fallback to local TTS. You MUST provide a clean WAV sample of your voice:

- Via config: Set `voice_profile_path` in config.yaml
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

- [System Overview](SYSTEM_OVERVIEW.md)
- [Quick Start Guide](QUICK_START.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Cross-Platform Packaging](CROSS_PLATFORM_PACKAGING.md)

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

Contributions are welcome! Please feel free to submit issues and pull requests.

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
