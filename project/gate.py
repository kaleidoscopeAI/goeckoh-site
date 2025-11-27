# backend/guardian/gate.py

from __future__ import annotations

from typing import Any, Dict

from backend.guardian.policy import GuardianPolicy
from backend.server import pending_guardian_suggestions # Import the list from server

def action_allowed(action: Dict[str, Any], policy: GuardianPolicy) -> bool:
    if not policy.environment_enabled:
        return False

    a_type = action.get("type")

    # HID branch
    if a_type == "hid":
        if not policy.hid_enabled:
            return False
        mode = policy.hid_mode()
        if mode == "observe":
            return False
        if mode == "suggest":
            queue_for_guardian_review(action)
            return False
        # mode == "auto"
        # You can add per-subtype checks here if you want
        return True

    # Smart device branch
    if a_type.startswith("smart_"):
        if not policy.smart_devices_enabled:
            return False
        mode = policy.smart_mode()
        if mode == "observe":
            return False
        if mode == "suggest":
            queue_for_guardian_review(action)
            return False
        # mode == "auto"
        # Example: lights level bounds check
        if a_type == "smart_light":
            level = float(action.get("level", 0.0))
            room = action.get("room", "bedroom")
            rooms = ((policy.raw.get("smart_devices") or {})
                     .get("lights", {})
                     .get("rooms", {}))
            cfg = rooms.get(room)
            if not cfg:
                return False
            if not (cfg.get("min_level", 0.0) <= level <= cfg.get("max_level", 1.0)):
                return False
        return True

    return False


def queue_for_guardian_review(action: Dict[str, Any]) -> None:
    # Append to the global list in server.py
    pending_guardian_suggestions.append(action)
