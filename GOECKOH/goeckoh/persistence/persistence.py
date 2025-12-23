# goeckoh/persistence.py

import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".goeckoh"
CONFIG_FILE = CONFIG_DIR / "config.json"
VOICES_FILE = CONFIG_DIR / "voices.json"

DEFAULT_CONFIG = {
    "active_voice_name": "Default Hybrid",
    "synthesis_mode": "hybrid",
}

DEFAULT_VOICES = [
    {
        "name": "Default Hybrid",
        "kind": "hybrid",
        "description": "Standard balanced voice"
    }
]

class PersistenceManager:
    """Handles loading and saving of app configuration and voice profiles."""

    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_default_files()

    def _ensure_default_files(self):
        """Create default config/voices files if they don't exist."""
        if not CONFIG_FILE.exists():
            self.save_config(DEFAULT_CONFIG)
        if not VOICES_FILE.exists():
            self.save_voices(DEFAULT_VOICES)

    def load_config(self) -> Dict[str, Any]:
        """Loads the main application configuration from config.json."""
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load config file, falling back to default: {e}")
            return DEFAULT_CONFIG.copy()

    def save_config(self, config: Dict[str, Any]):
        """Saves the configuration dictionary to config.json."""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save config file: {e}")

    def load_voices(self) -> List[Dict[str, Any]]:
        """Loads the list of voice profiles from voices.json."""
        try:
            with open(VOICES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load voices file, falling back to default: {e}")
            return DEFAULT_VOICES[:]

    def save_voices(self, voices: List[Dict[str, Any]]):
        """Saves the list of voice profiles to voices.json."""
        try:
            with open(VOICES_FILE, "w", encoding="utf-8") as f:
                json.dump(voices, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save voices file: {e}")

    def add_voice_profile(self, voice_data: Dict[str, Any]):
        """Adds a new voice profile to the voices file."""
        voices = self.load_voices()
        # Avoid duplicates
        if not any(v['name'] == voice_data['name'] for v in voices):
            voices.append(voice_data)
            self.save_voices(voices)