# Configuration Directory

This directory contains all system configuration files for the Goeckoh application.

## Configuration Files

### Main Configuration
- **`config.yaml`** - Main system configuration file in YAML format
  - System settings, paths, and options
  - Voice profile configuration
  - Audio device settings
  - Model paths and parameters
  
- **`config.json`** - JSON format configuration (alternative format)
  - Used by some components that prefer JSON
  - Generally contains similar settings to config.yaml

- **`config.schema.yaml`** - Configuration schema definition
  - Defines the structure and validation rules for config.yaml
  - Use with `python -m cli validate` to check configuration

### Validation and Tools
- **`config_validator.py`** - Configuration validation script
  - Validates configuration files against the schema
  - Provides helpful error messages

- **`config.py`** - Configuration utilities module
  - Helper functions for loading and managing configuration

### Platform-Specific Configuration
- **`buildozer.spec`** - Android/mobile build configuration
  - Buildozer specification for mobile app packaging
  - Platform-specific settings

- **`Bubble_Universe.desktop`** - Desktop launcher configuration (Linux)
- **`Goeckoh_System.desktop`** - System desktop launcher (Linux)

### Security and Policies
- **`guardian_policy.json`** - Safety and guardian policies
  - User safety monitoring settings
  - Intervention thresholds
  - Guardian/caregiver controls

### Deployment
- **`requirements_deployment.txt`** - Deployment-specific Python dependencies
  - Additional packages needed for production deployment

## Usage

### Validating Configuration
```bash
# From the repository root
python -m cli validate
```

### Auto-fixing Configuration Issues
```bash
# From the repository root
python -m cli fix
```

### Loading Configuration in Code
```python
# Python code can load configuration from this directory
from echo_core import load_config

# This will automatically look in config/config.json or config.json
config = load_config()  

# Or specify a path explicitly
config = load_config("config/config.yaml")
```

## Important Notes

1. **Backward Compatibility**: The system supports loading config files from both the `config/` directory and the root directory for backward compatibility.

2. **Symlinks**: Some modules may have symlinks to configuration files in this directory (e.g., `pipeline_core/config.yaml` → `config/config.yaml`)

3. **Version Control**: 
   - Configuration template files are tracked in git
   - User-specific configuration overrides should be added to `.gitignore`
   - Never commit secrets or API keys to configuration files

4. **Environment Variables**: Some sensitive configuration (API keys, credentials) should be set via environment variables rather than in configuration files.

## Configuration Locations

The system checks for configuration files in this order:
1. `config/config.yaml` or `config/config.json` (new location)
2. `config.yaml` or `config.json` in the root directory (backward compatibility)
3. Default configuration embedded in the code

## See Also

- Main README: [../README.md](../README.md)
- Documentation: [../docs/INDEX.md](../docs/INDEX.md)
- Quick Start Guide: [../docs/guides/QUICK_START.md](../docs/guides/QUICK_START.md)
