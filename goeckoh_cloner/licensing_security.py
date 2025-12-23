import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import LICENSE_FILE, LICENSE_PRODUCT_CODE


LICENSE_PATTERN = re.compile(rf"^{LICENSE_PRODUCT_CODE}-[A-Z0-9]{{4}}-[A-Z0-9]{{4}}$")


@dataclass
class LicenseInfo:
    key: str
    machine_id: str
    activated_at: float

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "machine_id": self.machine_id,
            "activated_at": self.activated_at,
        }

    @staticmethod
    def from_json(data: dict) -> "LicenseInfo":
        return LicenseInfo(
            key=data["key"],
            machine_id=data["machine_id"],
            activated_at=float(data["activated_at"]),
        )


def _get_machine_id() -> str:
    # Deterministic enough for offline lock: MAC-based integer -> hex
    return hex(uuid.getnode())[2:]


def is_plausible_key(key: str) -> bool:
    return bool(LICENSE_PATTERN.match(key.strip().upper()))


def load_license() -> Optional[LicenseInfo]:
    if not LICENSE_FILE.exists():
        return None
    try:
        with LICENSE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        lic = LicenseInfo.from_json(data)
        return lic
    except Exception:
        return None


def save_license(lic: LicenseInfo) -> None:
    LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LICENSE_FILE.open("w", encoding="utf-8") as f:
        json.dump(lic.to_json(), f, indent=2)


def check_license_valid() -> bool:
    lic = load_license()
    if lic is None:
        return False
    if not is_plausible_key(lic.key):
        return False
    # bind to machine
    if lic.machine_id != _get_machine_id():
        return False
    # could add expiry etc. For now, lifetime activation.
    return True


def activate_license(key: str) -> bool:
    """Validate format and bind to this machine. Returns True on success."""
    key = key.strip().upper()
    if not is_plausible_key(key):
        return False
    lic = LicenseInfo(
        key=key,
        machine_id=_get_machine_id(),
        activated_at=time.time(),
    )
    save_license(lic)
    return True
