# Goeckoh System Configuration Guide

## Overview

The Goeckoh system uses a multi-layered configuration approach with different files managing different aspects of the system. This document explains the configuration structure and how to properly configure the system.

## Configuration Files

### 1. `config.yaml` - Main System Configuration

**Purpose**: Primary configuration for the agent runtime, tools, and voice cloning settings.

**Key Sections**:
- `system_prompt`: The system prompt defining agent behavior
- `tools`: List of available tools/capabilities
- `tracing`: Enable/disable operation tracing
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `enable_voice_clone`: Whether to enable voice cloning feature
- `voice_profile_path`: Path to user's voice sample WAV file (required if voice cloning enabled)
- `documents_path`: Directory containing documents for the agent to read
- `min_voice_profile_duration`: Minimum required voice sample length in seconds

**Usage**:
```yaml
# Enable voice cloning
enable_voice_clone: true
voice_profile_path: "./voice_profiles/my_voice.wav"

# Set logging level
log_level: "INFO"

# Enable tracing
tracing: true
```

### 2. `config.json` - Audio Processing Configuration

**Purpose**: Low-level audio processing parameters, DSP settings, and model paths.

**Key Sections**:

#### Audio Settings
```json
{
  "sample_rate": 16000,    // Sample rate in Hz (16kHz recommended)
  "frame_size": 512,       // Audio frame size in samples
  "hop_size": 256,         // Hop size for STFT processing
  "history_len": 32        // Number of historical frames to keep
}
```

#### Voice Activity Detection (VAD) - `seg` section
```json
{
  "seg": {
    "theta_rms": 0.02,      // RMS energy threshold for voice detection
    "theta_zcr": 0.15,      // Zero-crossing rate threshold
    "min_voiced_run": 5     // Minimum consecutive voiced frames
  }
}
```

#### Speech-to-Text (ASR) - `asr` section
```json
{
  "asr": {
    "hidden_size": 64,
    "blank_id": 0,
    "vocab": " _abcdefghijklmnopqrstuvwxyz'",
    "weight_path": "project/echo_companion/JacksonCompanion/asr_weights.npz"
  }
}
```

#### Text-to-Speech (TTS) - `tts` section
```json
{
  "tts": {
    "embedding_dim": 64,
    "decoder_dim": 64,
    "griffin_lim_iters": 16,
    "max_duration": 5,
    "weight_path": "project/echo_companion/JacksonCompanion/tts_weights.npz"
  }
}
```

#### Crystalline Heart (Emotional System) - `crystal` section
```json
{
  "crystal": {
    "D": 24,           // Dimensionality of crystal lattice
    "F": 14,           // Number of feature channels
    "eta": 0.05,       // Learning rate
    "tau_low": 0.4,    // Lower coherence threshold
    "tau_high": 0.8,   // Upper coherence threshold
    "gamma": 1.2       // Scaling factor
  }
}
```

### 3. `rust_core/real_system_config.ini` - Production Runtime Configuration

**Purpose**: Runtime settings for production deployment including API, GUI, and safety parameters.

**Sections**:

#### [SYSTEM]
```ini
log_level = INFO          # Logging level
max_sessions = 100        # Maximum concurrent sessions
session_timeout = 3600    # Session timeout in seconds
backup_interval = 300     # Backup interval in seconds
auto_save = true          # Enable automatic saving
```

#### [AUDIO]
```ini
sample_rate = 16000       # Must match config.json (16000 Hz)
buffer_size = 1024        # Audio buffer size
input_device = default    # Microphone device
output_device = default   # Speaker device
channels = 1              # Mono audio
latency = low             # Latency mode (low/high)
```

#### [API]
```ini
host = localhost          # API server host
port = 8080               # API server port
enable_cors = true        # Enable CORS
rate_limit = 100          # Requests per minute
auth_required = false     # Require authentication
```

#### [GUI]
```ini
theme = dark              # UI theme (dark/light)
window_size = 1200x800    # Default window size
auto_start = true         # Start GUI automatically
minimize_to_tray = true   # Minimize to system tray
```

#### [SAFETY]
```ini
max_arousal = 0.9         # Maximum arousal level
stress_threshold = 0.8    # Stress detection threshold
emergency_stop = true     # Enable emergency stop
monitor_interval = 5      # Safety check interval (seconds)
```

#### [MEMORY]
```ini
max_memories = 10000      # Maximum stored memories
retention_days = 30       # Memory retention period
compression = true        # Compress stored memories
encryption = false        # Encrypt memories (set true for production)
```

## Environment-Specific Configurations

### Development (`config.dev.yaml`)

Optimized for rapid development and testing:
- **Voice cloning disabled** for faster startup
- **Debug logging** enabled
- **Mock devices** allowed
- **Hot reload** enabled
- **Fast mode** (reduced quality for speed)

**Usage**:
```bash
export GOECKOH_ENV=development
python -m cli start
```

### Production (`config.prod.yaml`)

Optimized for production deployment:
- **Voice cloning required**
- **Standard logging** (INFO level)
- **All models loaded**
- **Safety monitoring** enabled
- **Session persistence** enabled
- **Privacy protections** enforced

**Usage**:
```bash
export GOECKOH_ENV=production
python -m cli start
```

## Configuration Validation

### Running the Validator

```bash
python validate_config.py
```

This script checks:
- ✓ All configuration files exist
- ✓ Required fields are present
- ✓ File paths are valid
- ✓ Sample rates are consistent
- ✓ Model weights are accessible
- ✓ Subsystems are properly integrated

### Common Issues and Fixes

#### Issue: Voice profile not found
```
⚠️  Voice profile not found: ./example_voice_profile.wav
```

**Fix**: 
1. Create a voice profile directory: `mkdir -p voice_profiles`
2. Record your voice (5+ seconds): Use any audio recording software
3. Save as WAV: `voice_profiles/my_voice.wav`
4. Update config.yaml: `voice_profile_path: "./voice_profiles/my_voice.wav"`

OR disable voice cloning for development:
```yaml
enable_voice_clone: false
voice_profile_path: null
```

#### Issue: Inconsistent sample rates
```
⚠️  real_system_config.ini: Audio sample_rate is 22050, but config.json uses 16000
```

**Fix**: Update `rust_core/real_system_config.ini`:
```ini
[AUDIO]
sample_rate = 16000  # Match config.json
```

#### Issue: Model weights not found
```
⚠️  ASR weights not found: asr_weights.npz
```

**Fix**: Model weights are located in JacksonCompanion. The paths have been updated to:
```json
{
  "asr": {
    "weight_path": "project/echo_companion/JacksonCompanion/asr_weights.npz"
  },
  "tts": {
    "weight_path": "project/echo_companion/JacksonCompanion/tts_weights.npz"
  }
}
```

## Subsystem Integration

### CompleteUnifiedSystem
**Location**: `GOECKOH/goeckoh/systems/complete_unified_system.py`
**Configuration**: Uses `config.json` for audio processing parameters
**Dependencies**: Vosk (STT), Coqui TTS (voice synthesis)

### Cognitive Nebula
**Location**: `project/cognitive-nebula(8)/`
**Configuration**: `package.json`, environment variables
**Setup**:
```bash
cd project/cognitive-nebula\(8\)
npm install
npm run dev          # Development
npm run build        # Production build
npm run desktop      # Desktop app
```

### Goeckoh Web App
**Location**: `project/goeckoh/`
**Configuration**: `package.json`, `.env.local`
**Setup**:
```bash
cd project/goeckoh
npm install
# Set GEMINI_API_KEY in .env.local
npm run dev
```

## Best Practices

### 1. Sample Rate Consistency
**Always use 16kHz (16000 Hz)** across all configuration files:
- `config.json`: `"sample_rate": 16000`
- `rust_core/real_system_config.ini`: `sample_rate = 16000`
- JacksonCompanion: `"sample_rate": 16000`

### 2. Voice Profile Requirements
- **Minimum duration**: 5 seconds
- **Format**: WAV, 16-bit PCM
- **Sample rate**: 16kHz preferred
- **Clean audio**: No background noise
- **Single speaker**: Only your voice

### 3. Security for Production
- Enable encryption: `encryption = true` in `[MEMORY]` section
- Set `auth_required = true` in `[API]` section
- Use `config.prod.yaml` with strict validation
- Never commit voice profiles or user data to git

### 4. Path Management
- Use relative paths from repository root
- Keep voice profiles in `voice_profiles/` (gitignored)
- Model weights in `project/echo_companion/JacksonCompanion/`
- Documents in `documents/` directory

## Configuration Priority

When multiple configuration files exist, the system uses this priority:

1. Environment-specific config (`config.dev.yaml` or `config.prod.yaml`)
2. Main config (`config.yaml`)
3. Default values in code

## Troubleshooting

### Enable Debug Logging
```yaml
log_level: "DEBUG"
tracing: true
```

### Disable Voice Cloning (for testing)
```yaml
enable_voice_clone: false
voice_profile_path: null
```

### Run Health Check
```bash
python validate_config.py
```

### Check Subsystem Status
The validator automatically checks:
- CompleteUnifiedSystem availability
- Cognitive Nebula installation
- Goeckoh web app setup
- Model weight files
- Voice profile paths

## Support

For configuration issues:
1. Run `python validate_config.py` to identify problems
2. Check this documentation for solutions
3. Review error messages carefully
4. File an issue on GitHub with validation output

## Related Documentation

- [README.md](README.md) - System overview
- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) - Detailed system architecture
- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Production deployment
