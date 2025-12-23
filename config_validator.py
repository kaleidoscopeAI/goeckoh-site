"""
Configuration validation and auto-repair system

Provides comprehensive configuration validation with automatic
repair capabilities for the Bubble system.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import jsonschema

logger = logging.getLogger(__name__)

@dataclass
class ConfigIssue:
    """Configuration issue description"""
    severity: str  # "error", "warning", "info"
    field: str
    message: str
    auto_fixable: bool = False
    suggested_value: Any = None

class ConfigValidator:
    """Configuration validator with auto-repair capabilities"""
    
    def __init__(self, config_path: str, schema_path: str):
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.issues: List[ConfigIssue] = []
        self.logger = logging.getLogger(__name__)
        
    def load_schema(self) -> Dict[str, Any]:
        """Load configuration schema"""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except OSError as e:
            self.logger.error("Failed to load schema: %s", e)
            return {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load current configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except OSError as e:
            self.logger.error("Failed to load config: %s", e)
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            return True
        except OSError as e:
            self.logger.error("Failed to save config: %s", e)
            return False
    
    def validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate config against JSON schema"""
        try:
            jsonschema.validate(config, schema)
            return True
        except jsonschema.ValidationError as e:
            self.issues.append(ConfigIssue(
                severity="error",
                field=".".join(str(p) for p in e.absolute_path) if e.absolute_path else "root",
                message=str(e.message),
                auto_fixable=False
            ))
            return False
        except (jsonschema.SchemaError, TypeError) as e:
            self.logger.error("Schema validation error: %s", e)
            return False
    
    def validate_audio_config(self, config: Dict[str, Any]) -> None:
        """Validate audio-specific configuration"""
        audio_config = config.get('audio', {})
        
        # Sample rate validation
        sample_rate = audio_config.get('sample_rate', 16000)
        if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            self.issues.append(ConfigIssue(
                severity="warning",
                field="audio.sample_rate",
                message=f"Unusual sample rate: {sample_rate}",
                auto_fixable=True,
                suggested_value=16000
            ))
        
        # Buffer size validation
        buffer_size = audio_config.get('buffer_size', 1024)
        if buffer_size < 64 or buffer_size > 8192 or (buffer_size & (buffer_size - 1)) != 0:
            self.issues.append(ConfigIssue(
                severity="warning",
                field="audio.buffer_size",
                message=f"Invalid buffer size: {buffer_size}. Should be power of 2 between 64-8192",
                auto_fixable=True,
                suggested_value=1024
            ))
    
    def validate_model_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration"""
        models = config.get('models', {})
        
        # Check model file existence
        for model_type in ['tokens', 'encoder', 'decoder', 'tts']:
            model_path = models.get(model_type)
            if model_path and not Path(model_path).exists():
                self.issues.append(ConfigIssue(
                    severity="error",
                    field=f"models.{model_type}",
                    message=f"Model file not found: {model_path}",
                    auto_fixable=False
                ))
    
    def validate_ui_config(self, config: Dict[str, Any]) -> None:
        """Validate UI configuration"""
        ui_config = config.get('ui', {})
        
        # Bubble radius validation
        radius = ui_config.get('bubble_radius', 100.0)
        if radius < 10.0 or radius > 1000.0:
            self.issues.append(ConfigIssue(
                severity="warning",
                field="ui.bubble_radius",
                message=f"Unusual bubble radius: {radius}",
                auto_fixable=True,
                suggested_value=100.0
            ))
    
    def validate_voice_config(self, config: Dict[str, Any]) -> None:
        """Validate voice cloning configuration"""
        voice_config = config.get('voice', {})
        
        # Voice profile validation
        profile_path = voice_config.get('voice_profile_path')
        if profile_path and not Path(profile_path).exists():
            self.issues.append(ConfigIssue(
                severity="error",
                field="voice.voice_profile_path",
                message=f"Voice profile not found: {profile_path}",
                auto_fixable=False
            ))
        
        # Minimum duration validation
        min_duration = voice_config.get('min_voice_profile_duration', 5.0)
        if min_duration < 1.0 or min_duration > 60.0:
            self.issues.append(ConfigIssue(
                severity="warning",
                field="voice.min_voice_profile_duration",
                message=f"Unusual minimum duration: {min_duration}",
                auto_fixable=True,
                suggested_value=5.0
            ))
    
    def validate_all(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[ConfigIssue]:
        """Run all validation checks"""
        self.issues.clear()
        
        # Schema validation
        self.validate_schema(config, schema)
        
        # Component-specific validation
        self.validate_audio_config(config)
        self.validate_model_config(config)
        self.validate_ui_config(config)
        self.validate_voice_config(config)
        
        return self.issues
    
    def auto_fix_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automatic fixes to configuration"""
        fixed_config = config.copy()
        fixes_applied = []
        
        for issue in self.issues:
            if issue.auto_fixable and issue.suggested_value is not None:
                # Navigate to the field and apply fix

                field_parts = issue.field.split('.')
                current = fixed_config
                
                # Navigate to parent of target field
                for part in field_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Apply the fix
                current[field_parts[-1]] = issue.suggested_value
                fixes_applied.append(f"Fixed {issue.field}: {issue.suggested_value}")
        
        if fixes_applied:
            self.logger.info("Applied %d automatic fixes", len(fixes_applied))
            for fix in fixes_applied:
                self.logger.info("  - %s", fix)
        
        return fixed_config
