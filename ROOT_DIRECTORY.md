# Root Directory Contents

This document lists all files and directories at the repository root, explaining their purpose.

## Files at Root

### Documentation
- **`README.md`** - Main repository documentation and getting started guide
- **`CONTRIBUTING.md`** - Guidelines for contributing to the project
- **`REORGANIZATION_SUMMARY.md`** - Complete summary of the repository reorganization

### Configuration
- **`requirements.txt`** - Python dependencies for the main application
- **`.gitignore`** - Git ignore patterns

### Core Application Files
The following Python files are core to the application and remain at root for compatibility:
- `__init__.py` - Package initialization
- `echo_core.py` - Core echo system functionality
- `main_app.py` - Main application entry point
- `system_launcher.py` - System launcher
- `system_monitor.py` - System monitoring
- `test_system.py` - System tests
- `kaleidoscope_ai.py` - Kaleidoscope AI integration
- `realtime_voice_pipeline.py` - Real-time voice processing
- Various module files (audio, behavior, voice, etc.)

## Directories at Root

### Main Directories
- **`website/`** - Marketing and information website (self-contained)
- **`docs/`** - All documentation (see [docs/INDEX.md](docs/INDEX.md))
- **`config/`** - Configuration files (see [config/README.md](config/README.md))
- **`scripts/`** - Build and deployment scripts (see [scripts/README.md](scripts/README.md))
- **`archive/`** - Archived materials (see [archive/README.md](archive/README.md))

### Application Components
- **`GOECKOH/`** - Main application package
- **`cognitive-nebula/`** - 3D visualization frontend (React + Three.js)
- **`mobile/`** - Mobile platform code (iOS/Android)
- **`frontend/`** - Frontend components
- **`rust_core/`** - Rust performance core

### Supporting Directories
- **`assets/`** - Application assets (models, icons, etc.)
- **`tests/`** - Test files
- **`examples/`** - Example code and usage
- **`data/`** - Data files

### Module Directories
- **`src/`** - Source code
- **`apps/`** - Application modules
- **`systems/`** - System components
- **`pipeline_core/`** - Core pipeline functionality
- **`goeckoh_cloner/`** - Voice cloning system
- **`integrations/`** - External integrations
- **`persistence/`** - Data persistence layer
- **`audio/`**, **`voice/`**, **`heart/`**, **`icons/`** - Component directories

### Build and Project Directories
- **`project/`** - Project files and configurations
- **`web_assets/`** - Web assets
- **`.git/`** - Git version control (hidden)

## Organization Principles

1. **Clean Root** - Only essential files at root level
2. **Logical Grouping** - Related files in appropriate directories
3. **Clear Documentation** - Each major directory has a README
4. **Easy Navigation** - Consistent structure throughout

## See Also

- [Repository Structure](README.md#repository-structure)
- [Documentation Index](docs/INDEX.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Reorganization Summary](REORGANIZATION_SUMMARY.md)

---

Last updated: December 24, 2024
