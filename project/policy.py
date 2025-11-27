# backend/guardian/policy.py

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class GuardianPolicy:
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "GuardianPolicy":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            # Provide a default minimal policy if file doesn't exist
            data = {
                "version": 1,
                "child_id": "default",
                "profile_name": "Default Profile",
                "environment_enabled": True,
                "echo": {"corrections_enabled": True, "max_corrections_per_minute": 12, "min_gap_between_corrections_sec": 2.0, "allow_calming_scripts": True, "allow_social_stories": True, "night_mode_quiet_hours": {"enabled": False, "start_local": "20:00", "end_local": "07:00", "max_volume": 0.4}},
                "hid": {"enabled": True, "mode": "suggest", "allowed_actions": ["keyboard_type", "keyboard_combo", "mouse_move", "mouse_click"], "rate_limit_per_minute": 10},
                "smart_devices": {"enabled": True, "mode": "suggest", "lights": {"enabled": True, "rooms": {"bedroom": {"min_level": 0.2, "max_level": 0.7, "allow_color_temp_change": True}}}, "media": {"enabled": True, "allow_pause_play": True, "allow_mute": True, "allow_content_change": False}},
                "modes": {"baseline": {"label": "Everyday", "description": "Normal operation."}},
                "active_mode": "baseline",
                "logging": {"log_environment_actions": True, "log_details": "summary", "retain_days": 30}
            }
        return cls(raw=data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.raw, indent=2), encoding="utf-8")

    # Convenience properties
    @property
    def environment_enabled(self) -> bool:
        return bool(self.raw.get("environment_enabled", False))

    @property
    def hid_enabled(self) -> bool:
        active_mode = self.raw.get("active_mode", "baseline")
        mode_cfg = (self.raw.get("modes") or {}).get(active_mode, {})
        override = mode_cfg.get("hid_enabled_override")
        base = bool((self.raw.get("hid") or {}).get("enabled", False))
        return bool(base if override is None else override)

    @property
    def smart_devices_enabled(self) -> bool:
        active_mode = self.raw.get("active_mode", "baseline")
        mode_cfg = (self.raw.get("modes") or {}).get(active_mode, {})
        override = mode_cfg.get("smart_devices_enabled_override")
        base = bool((self.raw.get("smart_devices") or {}).get("enabled", False))
        return bool(base if override is None else override)

    def hid_mode(self) -> str:
        return (self.raw.get("hid") or {}).get("mode", "observe")

    def smart_mode(self) -> str:
        return (self.raw.get("smart_devices") or {}).get("mode", "observe")
