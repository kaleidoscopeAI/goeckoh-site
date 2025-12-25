#!/usr/bin/env python3
"""
Configuration Validation Script for Goeckoh System

This script validates all configuration files across the system,
checks for consistency, and reports any issues or missing paths.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import configparser

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. YAML validation will be limited.")


class ConfigValidator:
    """Validates Goeckoh system configuration files."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("=" * 70)
        print("Goeckoh System Configuration Validator")
        print("=" * 70)
        print()
        
        # Validate main configuration files
        self.validate_yaml_config()
        self.validate_json_config()
        self.validate_ini_config()
        self.validate_jackson_companion_config()
        
        # Check subsystem integration
        self.check_subsystem_integration()
        
        # Check model paths
        self.check_model_paths()
        
        # Check voice profile paths
        self.check_voice_profiles()
        
        # Print results
        self.print_results()
        
        return len(self.errors) == 0
    
    def validate_yaml_config(self):
        """Validate config.yaml."""
        config_path = self.base_path / "config.yaml"
        schema_path = self.base_path / "config.schema.yaml"
        
        if not config_path.exists():
            self.errors.append(f"Missing config.yaml at {config_path}")
            return
        
        self.info.append(f"✓ Found config.yaml")
        
        if not YAML_AVAILABLE:
            self.warnings.append("Cannot validate YAML structure (PyYAML not installed)")
            return
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            if "system_prompt" not in config:
                self.errors.append("config.yaml: Missing 'system_prompt'")
            
            if "tools" not in config:
                self.errors.append("config.yaml: Missing 'tools'")
            
            # Check voice cloning configuration
            if config.get("enable_voice_clone", False):
                voice_path = config.get("voice_profile_path")
                if not voice_path:
                    self.errors.append("config.yaml: voice_profile_path required when enable_voice_clone is true")
                else:
                    full_path = self.base_path / voice_path
                    if not full_path.exists():
                        self.warnings.append(f"config.yaml: voice_profile_path points to non-existent file: {voice_path}")
            
            # Validate log level
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            log_level = config.get("log_level", "INFO")
            if log_level not in valid_log_levels:
                self.errors.append(f"config.yaml: Invalid log_level '{log_level}'. Must be one of {valid_log_levels}")
            
            self.info.append(f"✓ config.yaml structure is valid")
            
        except yaml.YAMLError as e:
            self.errors.append(f"config.yaml: YAML parsing error: {e}")
        except Exception as e:
            self.errors.append(f"config.yaml: Validation error: {e}")
    
    def validate_json_config(self):
        """Validate config.json."""
        config_path = self.base_path / "config.json"
        
        if not config_path.exists():
            self.errors.append(f"Missing config.json at {config_path}")
            return
        
        self.info.append(f"✓ Found config.json")
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Check required audio processing fields
            required_fields = ["sample_rate", "frame_size", "hop_size"]
            for field in required_fields:
                if field not in config:
                    self.errors.append(f"config.json: Missing required field '{field}'")
            
            # Check subsections
            required_sections = ["routine", "seg", "text", "asr", "tts", "crystal"]
            for section in required_sections:
                if section not in config:
                    self.warnings.append(f"config.json: Missing recommended section '{section}'")
            
            # Validate sample rate consistency
            sample_rate = config.get("sample_rate")
            if sample_rate and sample_rate != 16000:
                self.warnings.append(f"config.json: Non-standard sample_rate {sample_rate} (recommended: 16000)")
            
            # Check for model weight paths
            if "asr" in config and "weight_path" in config["asr"]:
                asr_weights = self.base_path / config["asr"]["weight_path"]
                if not asr_weights.exists():
                    self.warnings.append(f"config.json: ASR weight_path not found: {config['asr']['weight_path']}")
            
            if "tts" in config and "weight_path" in config["tts"]:
                tts_weights = self.base_path / config["tts"]["weight_path"]
                if not tts_weights.exists():
                    self.warnings.append(f"config.json: TTS weight_path not found: {config['tts']['weight_path']}")
            
            self.info.append(f"✓ config.json structure is valid")
            
        except json.JSONDecodeError as e:
            self.errors.append(f"config.json: JSON parsing error: {e}")
        except Exception as e:
            self.errors.append(f"config.json: Validation error: {e}")
    
    def validate_ini_config(self):
        """Validate rust_core/real_system_config.ini."""
        config_path = self.base_path / "rust_core" / "real_system_config.ini"
        
        if not config_path.exists():
            self.warnings.append(f"rust_core/real_system_config.ini not found (optional)")
            return
        
        self.info.append(f"✓ Found rust_core/real_system_config.ini")
        
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Check required sections
            required_sections = ["SYSTEM", "AUDIO", "API", "GUI", "SAFETY", "MEMORY"]
            for section in required_sections:
                if section not in config:
                    self.warnings.append(f"real_system_config.ini: Missing section [{section}]")
            
            # Validate audio sample rate consistency
            if "AUDIO" in config:
                ini_sample_rate = config.get("AUDIO", "sample_rate", fallback=None)
                if ini_sample_rate and int(ini_sample_rate) != 16000:
                    self.warnings.append(
                        f"real_system_config.ini: Audio sample_rate is {ini_sample_rate}, "
                        f"but config.json uses 16000. Consider using consistent values."
                    )
            
            self.info.append(f"✓ real_system_config.ini structure is valid")
            
        except Exception as e:
            self.errors.append(f"real_system_config.ini: Validation error: {e}")
    
    def validate_jackson_companion_config(self):
        """Validate project/echo_companion/JacksonCompanion/config.json."""
        config_path = self.base_path / "project" / "echo_companion" / "JacksonCompanion" / "config.json"
        
        if not config_path.exists():
            self.info.append("JacksonCompanion config not found (optional subsystem)")
            return
        
        self.info.append(f"✓ Found JacksonCompanion/config.json")
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Check consistency with main config.json
            if "audio" in config:
                jc_sample_rate = config["audio"].get("sample_rate")
                if jc_sample_rate and jc_sample_rate != 16000:
                    self.warnings.append(
                        f"JacksonCompanion: sample_rate is {jc_sample_rate}, "
                        f"but main config uses 16000"
                    )
            
            self.info.append(f"✓ JacksonCompanion/config.json is valid")
            
        except json.JSONDecodeError as e:
            self.errors.append(f"JacksonCompanion/config.json: JSON parsing error: {e}")
        except Exception as e:
            self.errors.append(f"JacksonCompanion/config.json: Validation error: {e}")
    
    def check_subsystem_integration(self):
        """Check if subsystems are properly integrated."""
        # Check for CompleteUnifiedSystem
        unified_system = self.base_path / "GOECKOH" / "goeckoh" / "systems" / "complete_unified_system.py"
        if unified_system.exists():
            self.info.append("✓ CompleteUnifiedSystem found")
        else:
            self.warnings.append("CompleteUnifiedSystem not found at expected location")
        
        # Check for Cognitive Nebula
        cognitive_nebula = self.base_path / "project" / "cognitive-nebula(8)"
        if cognitive_nebula.exists():
            self.info.append("✓ Cognitive Nebula subsystem found")
            # Check for package.json
            pkg_json = cognitive_nebula / "package.json"
            if pkg_json.exists():
                self.info.append("  ✓ Cognitive Nebula has package.json")
            else:
                self.warnings.append("  Cognitive Nebula missing package.json")
        else:
            self.warnings.append("Cognitive Nebula subsystem not found")
        
        # Check for Goeckoh web app
        goeckoh_app = self.base_path / "project" / "goeckoh"
        if goeckoh_app.exists():
            self.info.append("✓ Goeckoh web app found")
            pkg_json = goeckoh_app / "package.json"
            if pkg_json.exists():
                self.info.append("  ✓ Goeckoh web app has package.json")
            else:
                self.warnings.append("  Goeckoh web app missing package.json")
        else:
            self.warnings.append("Goeckoh web app not found")
    
    def check_model_paths(self):
        """Check if model weight files exist."""
        # Check main config.json model paths
        config_path = self.base_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # ASR weights
            if "asr" in config and "weight_path" in config["asr"]:
                weight_path = self.base_path / config["asr"]["weight_path"]
                if weight_path.exists():
                    self.info.append(f"✓ ASR weights found: {config['asr']['weight_path']}")
                else:
                    self.warnings.append(f"ASR weights not found: {config['asr']['weight_path']}")
            
            # TTS weights
            if "tts" in config and "weight_path" in config["tts"]:
                weight_path = self.base_path / config["tts"]["weight_path"]
                if weight_path.exists():
                    self.info.append(f"✓ TTS weights found: {config['tts']['weight_path']}")
                else:
                    self.warnings.append(f"TTS weights not found: {config['tts']['weight_path']}")
        
        # Check JacksonCompanion weights
        jc_dir = self.base_path / "project" / "echo_companion" / "JacksonCompanion"
        if jc_dir.exists():
            for weight_file in ["asr_weights.npz", "tts_weights.npz", "crystal_weights.npz"]:
                weight_path = jc_dir / weight_file
                if weight_path.exists():
                    self.info.append(f"✓ JacksonCompanion {weight_file} found")
                else:
                    self.warnings.append(f"JacksonCompanion {weight_file} not found")
    
    def check_voice_profiles(self):
        """Check voice profile configuration."""
        config_path = self.base_path / "config.yaml"
        
        if not config_path.exists() or not YAML_AVAILABLE:
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if config.get("enable_voice_clone"):
            voice_path = config.get("voice_profile_path", "")
            full_path = self.base_path / voice_path
            
            if full_path.exists():
                self.info.append(f"✓ Voice profile exists: {voice_path}")
            else:
                self.warnings.append(
                    f"Voice profile not found: {voice_path}\n"
                    f"  Create a voice profile or disable voice cloning in config.yaml"
                )
    
    def print_results(self):
        """Print validation results."""
        print()
        print("=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print()
        
        if self.info:
            print("ℹ️  Information:")
            for msg in self.info:
                print(f"  {msg}")
            print()
        
        if self.warnings:
            print("⚠️  Warnings:")
            for msg in self.warnings:
                print(f"  {msg}")
            print()
        
        if self.errors:
            print("❌ Errors:")
            for msg in self.errors:
                print(f"  {msg}")
            print()
        
        print("=" * 70)
        if self.errors:
            print("❌ VALIDATION FAILED")
            print(f"   {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        elif self.warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            print(f"   {len(self.warnings)} warning(s)")
        else:
            print("✅ VALIDATION PASSED")
            print("   All configuration files are valid!")
        print("=" * 70)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.resolve()
    
    validator = ConfigValidator(base_path)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
