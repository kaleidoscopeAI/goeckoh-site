# GitHub Copilot Instructions for Goeckoh

## Project Overview

**Goeckoh (Go Echo)** is an offline-first speech support system designed specifically for neurodivergent individuals (particularly those with autism). It provides real-time voice feedback, speech correction, and therapeutic interventions through an advanced AI system.

### Key Features
- Identity-matched voice replay using the user's own voice
- Offline-first architecture for privacy and reliability
- Real-time speech correction with 6-step voice processing pipeline
- Autism-optimized features (VAD with 1.2s pause tolerance, ABA therapeutics)
- Emotional regulation system (Crystalline Heart - 1024-node lattice)
- Hybrid Python/Rust architecture for performance

### Repository Structure
- `src/`, `goeckoh_cloner/`, `pipeline_core/` - Application source code (Python)
- Root directory `*.jsx` files - React/Three.js visualization components
- `frontend/goeckoh-react/` - React frontend project
- `website/` - Marketing and information website (static HTML/CSS/JS)
- `tests/` - Test files (pytest)
- `docs/` - Documentation
- `assets/` - Application assets (models, etc.)

## Code Style and Conventions

### Python
- **Use Python 3.8+ features**
- **Follow PEP 8 style guide** for code formatting
- **Use type hints** where appropriate for function signatures
- **Docstrings**: Use triple-quoted strings for module, class, and function documentation
- **Imports**: Standard library first, then third-party, then local imports
- **Error handling**: Use specific exception types, not bare `except:`
- **Testing**: Use pytest for all tests, follow `test_*.py` naming convention

### JavaScript/React
- **Use ES6+ modern JavaScript**
- **Use JSX for React components**
- **Single quotes** for strings (when not using template literals)
- **Semicolons**: End statements with semicolons
- **Components**: Prefer functional components with hooks over class components
- **File naming**: Use PascalCase for component files (e.g., `App.jsx`, `KaleidoscopeCanvas.jsx`)

### HTML/CSS
- **Use semantic HTML5** elements
- **Follow BEM naming** for CSS classes where applicable
- **Mobile-first approach** for responsive design
- **Accessibility**: Include ARIA labels and proper semantic structure

## Testing

### Python Tests
- **Framework**: Use pytest for all Python tests
- **Location**: Place tests in the `tests/` directory
- **Naming**: Test files must start with `test_` (e.g., `test_config_validation.py`)
- **Coverage**: Write tests for new functionality and bug fixes
- **Fixtures**: Use pytest fixtures for common test setup
- **Running tests**: Execute with `python -m pytest` or `tests/test_runner.py`

### Test Example Pattern
```python
import pytest
from pathlib import Path

def test_feature_validates():
    # Arrange
    config = load_config()
    
    # Act
    result = validate_feature(config)
    
    # Assert
    assert result is True
```

## Security Best Practices

### Privacy-First Architecture
- **Never send audio data to external servers** - all processing must be local
- **No hardcoded credentials** - use environment variables or config files
- **Validate all user input** - especially file paths and audio data
- **Secure voice cloning** - protect voice profile data as sensitive PII
- **Local-only TTS**: Use Coqui TTS and other offline models

### Specific Security Rules
- **Audio files**: Always validate format, duration, and sample rate before processing
- **File operations**: Use `Path` from `pathlib` for safe path manipulation
- **Config validation**: Validate all configuration against schema before use
- **Dependencies**: Keep dependencies minimal and audited (especially for audio processing)

## Project-Specific Guidelines

### Voice Processing
- **Minimum voice sample duration**: 5 seconds for voice cloning
- **Preferred audio format**: 16kHz WAV
- **VAD (Voice Activity Detection)**: Use 1.2 second pause tolerance for autism-optimized behavior
- **Latency requirements**: Fast path must maintain < 100ms latency for therapeutic mirroring

### Configuration
- **Config file**: Use `config.yaml` for application configuration
- **Schema validation**: Always validate against `config.schema.yaml`
- **CLI commands**: 
  - `python -m cli validate` - Validate config
  - `python -m cli fix` - Auto-fix common issues
  - `python -m cli start` - Start REPL

### Build and Development
- **Python setup**: 
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- **Frontend development**:
  ```bash
  npm install
  npm run dev    # Development server
  npm run build  # Production build
  ```
- **Testing**: Run `python -m pytest` for Python tests

### Documentation
- **README.md**: Main project documentation
- **SYSTEM_OVERVIEW.md**: Detailed system architecture and functionality
- **QUICK_START.md**: Quick start guide for users
- **DEPLOYMENT_GUIDE.md**: Deployment instructions
- Update documentation when adding new features or changing architecture

## Common Tasks

### Adding a New Python Module
1. Create the module in the appropriate directory (`src/`, `goeckoh_cloner/`, etc.)
2. Add docstrings for all public functions and classes
3. Add type hints to function signatures
4. Create corresponding test file in `tests/` directory
5. Import and use the module in relevant components
6. Update documentation if it's a user-facing feature

### Adding a New React Component
1. Create component file in PascalCase (e.g., `MyComponent.jsx`)
2. Use functional components with hooks
3. Import required dependencies at the top
4. Export component as default export
5. Use the component in parent components as needed

### Modifying Voice Processing Pipeline
1. **Understand the 6-step pipeline**: Audio Capture → STT → Linguistic Correction → TTS → Prosody Transfer → Playback
2. **Maintain latency requirements**: Fast path < 100ms
3. **Test with real audio samples**: Minimum 5s, 16kHz WAV
4. **Validate autism-optimized features**: Ensure VAD tolerance and pause handling work correctly
5. **Update documentation**: Reflect changes in SYSTEM_OVERVIEW.md

## Dependencies

### Python Dependencies
- **Core**: numpy, scipy, sounddevice, Cython
- **Voice processing**: sherpa-onnx
- **UI**: kivy, kivymd, textual
- **Network**: requests
- **Testing**: pytest (dev dependency)
- **Optional**: TTS (Coqui TTS for voice cloning - listed in requirements_deployment.txt)
- Only add new dependencies if absolutely necessary and after security review

### JavaScript Dependencies
- **React**: react, react-dom (^19.1.1 in root, ^19.2.0 in frontend/goeckoh-react)
- **3D Graphics**: three, @react-three/fiber, @react-three/drei
- **Build**: vite, @vitejs/plugin-react
- **TypeScript**: For type checking where needed
- Prefer existing dependencies over adding new ones

## Accessibility

- **ARIA labels**: Always include for interactive elements
- **Keyboard navigation**: Ensure all features are keyboard accessible
- **Screen readers**: Use semantic HTML and ARIA attributes
- **Autism-friendly design**: Clear, predictable UI with minimal sensory overload
- **Focus management**: Proper focus indicators and logical tab order

## Best Practices for Copilot Agent

### When Working on Issues
- **Read the full context**: Review related files and documentation before making changes
- **Minimal changes**: Make the smallest possible changes to address the issue
- **Test thoroughly**: Run existing tests and add new ones for your changes
- **Document changes**: Update relevant documentation
- **Security check**: Ensure changes don't introduce vulnerabilities (especially around audio processing and file handling)

### When Reviewing Code
- **Privacy concerns**: Flag any code that might send data externally
- **Performance**: Watch for blocking operations in the voice processing pipeline
- **Error handling**: Ensure proper error handling, especially for audio file operations
- **Type safety**: Check for type hints in Python, proper typing in TypeScript

### When Adding Features
- **Offline-first**: All features must work without internet connection
- **Privacy-forward**: No data collection or external API calls without explicit user consent
- **Autism-optimized**: Consider sensory sensitivities and processing differences
- **Performance**: Maintain real-time performance requirements (< 100ms for fast path)
