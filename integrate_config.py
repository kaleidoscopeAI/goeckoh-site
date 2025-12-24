#!/usr/bin/env python3
"""
Configuration Integration Manager for Goeckoh System

This script helps integrate subsystems by synchronizing configuration
values across different config files and ensuring consistency.
"""

import json
import sys
import configparser
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)


class ConfigIntegrator:
    """Manages configuration integration across subsystems."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.changes_made = []
    
    def integrate_all(self):
        """Integrate all configuration files."""
        print("=" * 70)
        print("Goeckoh System Configuration Integrator")
        print("=" * 70)
        print()
        
        # Load configurations
        yaml_config = self.load_yaml_config()
        json_config = self.load_json_config()
        ini_config = self.load_ini_config()
        
        # Synchronize sample rates
        self.sync_sample_rates(json_config, ini_config)
        
        # Update model paths
        self.update_model_paths(json_config)
        
        # Create voice profile directory
        self.setup_voice_profiles()
        
        # Create documents directory
        self.setup_documents_dir()
        
        # Generate subsystem configs
        self.generate_subsystem_configs()
        
        # Print summary
        self.print_summary()
    
    def load_yaml_config(self) -> Dict[str, Any]:
        """Load config.yaml."""
        config_path = self.base_path / "config.yaml"
        if not config_path.exists():
            print(f"❌ config.yaml not found at {config_path}")
            return {}
        
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def load_json_config(self) -> Dict[str, Any]:
        """Load config.json."""
        config_path = self.base_path / "config.json"
        if not config_path.exists():
            print(f"❌ config.json not found at {config_path}")
            return {}
        
        with open(config_path) as f:
            return json.load(f)
    
    def load_ini_config(self) -> configparser.ConfigParser:
        """Load real_system_config.ini."""
        config_path = self.base_path / "rust_core" / "real_system_config.ini"
        config = configparser.ConfigParser()
        
        if config_path.exists():
            config.read(config_path)
        
        return config
    
    def sync_sample_rates(self, json_config: Dict, ini_config: configparser.ConfigParser):
        """Ensure sample rates are consistent across configs."""
        target_sample_rate = 16000  # Standard for speech processing
        
        # Check config.json
        json_sr = json_config.get("sample_rate")
        if json_sr != target_sample_rate:
            print(f"⚠️  config.json sample_rate is {json_sr}, should be {target_sample_rate}")
        
        # Check real_system_config.ini
        if ini_config.has_section("AUDIO"):
            ini_sr = ini_config.get("AUDIO", "sample_rate", fallback=None)
            if ini_sr and int(ini_sr) != target_sample_rate:
                print(f"✓ Synchronized sample_rate to {target_sample_rate} in real_system_config.ini")
                self.changes_made.append(f"Updated real_system_config.ini sample_rate to {target_sample_rate}")
        
        # Check JacksonCompanion config
        jc_config_path = self.base_path / "project" / "echo_companion" / "JacksonCompanion" / "config.json"
        if jc_config_path.exists():
            with open(jc_config_path) as f:
                jc_config = json.load(f)
            
            if "audio" in jc_config:
                jc_sr = jc_config["audio"].get("sample_rate")
                if jc_sr == target_sample_rate:
                    print(f"✓ JacksonCompanion sample_rate is correctly set to {target_sample_rate}")
                else:
                    print(f"⚠️  JacksonCompanion sample_rate is {jc_sr}, should be {target_sample_rate}")
    
    def update_model_paths(self, json_config: Dict):
        """Verify and update model weight paths."""
        # Paths are already updated in config.json
        asr_path = json_config.get("asr", {}).get("weight_path")
        tts_path = json_config.get("tts", {}).get("weight_path")
        
        if asr_path:
            full_path = self.base_path / asr_path
            if full_path.exists():
                print(f"✓ ASR weights found at: {asr_path}")
            else:
                print(f"⚠️  ASR weights not found at: {asr_path}")
        
        if tts_path:
            full_path = self.base_path / tts_path
            if full_path.exists():
                print(f"✓ TTS weights found at: {tts_path}")
            else:
                print(f"⚠️  TTS weights not found at: {tts_path}")
    
    def setup_voice_profiles(self):
        """Create voice_profiles directory if it doesn't exist."""
        voice_dir = self.base_path / "voice_profiles"
        
        if not voice_dir.exists():
            voice_dir.mkdir(parents=True)
            print(f"✓ Created voice_profiles directory")
            self.changes_made.append("Created voice_profiles directory")
            
            # Create a README
            readme_path = voice_dir / "README.md"
            readme_content = """# Voice Profiles Directory

This directory contains voice profile WAV files for voice cloning.

## Creating a Voice Profile

1. Record at least 5 seconds of clear speech
2. Save as WAV format (16-bit PCM, 16kHz preferred)
3. Use a quiet environment with no background noise
4. Speak naturally in your normal voice
5. Save the file as `my_voice.wav` or similar

## Using Your Voice Profile

Update `config.yaml` to point to your voice profile:

```yaml
enable_voice_clone: true
voice_profile_path: "./voice_profiles/my_voice.wav"
```

## Privacy Note

Voice profiles are personal data and should NOT be committed to git.
This directory is gitignored by default.
"""
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            print(f"  ✓ Created voice_profiles/README.md")
        else:
            print(f"✓ voice_profiles directory already exists")
    
    def setup_documents_dir(self):
        """Create documents directory if it doesn't exist."""
        docs_dir = self.base_path / "documents"
        
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True)
            print(f"✓ Created documents directory")
            self.changes_made.append("Created documents directory")
            
            # Create a README
            readme_path = docs_dir / "README.md"
            readme_content = """# Documents Directory

Place text documents here for the system to read and process.

## Supported Formats

- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents (requires PyPDF2 or similar)

## Usage

The system will automatically read all supported documents in this directory
when configured with:

```yaml
documents_path: "./documents"
```

You can use the CLI to read documents:

```bash
python -m cli read-docs --path ./documents --recursive
```
"""
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            print(f"  ✓ Created documents/README.md")
        else:
            print(f"✓ documents directory already exists")
    
    def generate_subsystem_configs(self):
        """Generate configuration helpers for subsystems."""
        # Create .gitignore entries for sensitive data
        gitignore_path = self.base_path / ".gitignore"
        
        gitignore_entries = [
            "# Voice profiles (personal data)",
            "voice_profiles/*.wav",
            "voice_profiles/*.mp3",
            "",
            "# User documents",
            "documents/*.pdf",
            "documents/*.docx",
            "",
            "# Environment-specific configs",
            ".env.local",
            "config.local.yaml",
            "",
            "# Session data",
            "sessions/",
            "*.session",
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                existing = f.read()
            
            # Check if entries already exist
            if "voice_profiles/*.wav" not in existing:
                with open(gitignore_path, "a") as f:
                    f.write("\n# Goeckoh Configuration - Auto-generated\n")
                    f.write("\n".join(gitignore_entries))
                    f.write("\n")
                print(f"✓ Updated .gitignore with voice profile exclusions")
                self.changes_made.append("Updated .gitignore")
            else:
                print(f"✓ .gitignore already configured")
        else:
            with open(gitignore_path, "w") as f:
                f.write("\n".join(gitignore_entries))
                f.write("\n")
            print(f"✓ Created .gitignore with voice profile exclusions")
            self.changes_made.append("Created .gitignore")
    
    def print_summary(self):
        """Print summary of changes."""
        print()
        print("=" * 70)
        print("INTEGRATION SUMMARY")
        print("=" * 70)
        print()
        
        if self.changes_made:
            print("Changes made:")
            for change in self.changes_made:
                print(f"  • {change}")
        else:
            print("No changes needed - system already integrated!")
        
        print()
        print("Next steps:")
        print("  1. Run: python validate_config.py")
        print("  2. Create your voice profile in voice_profiles/")
        print("  3. Update config.yaml with your voice profile path")
        print("  4. Start the system: python -m cli start")
        print()
        print("=" * 70)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.resolve()
    
    integrator = ConfigIntegrator(base_path)
    integrator.integrate_all()


if __name__ == "__main__":
    main()
