"""
Goeckoh License Client — embed this in the app (Python desktop version).
Android version uses the same logic in Kotlin (see GoeckohViewModel.kt).

Usage in the main app startup:
    from license_client import LicenseClient, LicenseState

    client = LicenseClient()
    state = client.check()
    if state == LicenseState.VALID:
        # proceed normally
    elif state == LicenseState.GRACE_PERIOD:
        # show "payment failed" banner, continue running
    else:
        # disable synthesis, show renewal UI
"""

import os
import json
import time
import uuid
import hmac as _hmac
import hashlib
import platform
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import jwt
import requests

log = logging.getLogger("goeckoh.license_client")

LICENSE_SERVER = os.environ.get("GOECKOH_LICENSE_SERVER", "https://api.goeckoh.com")
TOKEN_CACHE_PATH = Path.home() / ".goeckoh" / "license_token.json"
LICENSE_KEY_PATH = Path.home() / ".goeckoh" / "license_key.txt"
DEVICE_ID_PATH   = Path.home() / ".goeckoh" / "device_id.dat"
REQUEST_TIMEOUT  = 5.0
# JWT_PUBLIC_SECRET intentionally removed — the client does not verify
# JWT signatures.  Token integrity is protected by a device-specific HMAC
# on the cache file instead (see _save_token / _load_cached_token).


class LicenseState(str, Enum):
    VALID = "valid"
    GRACE_PERIOD = "grace_period"
    EXPIRED = "expired"
    NOT_ACTIVATED = "not_activated"
    OFFLINE_VALID = "offline_valid"
    OFFLINE_EXPIRED = "offline_expired"


def _device_fingerprint() -> str:
    """
    Stable per-device identifier.  Not personally identifiable —
    hashes OS-level machine info plus a persistent random component so
    MAC-address randomization cannot shift the fingerprint between boots.
    Never sent off-device raw.
    """
    # Persistent random component: written once on first launch
    if not DEVICE_ID_PATH.exists():
        DEVICE_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEVICE_ID_PATH.write_text(uuid.uuid4().hex)
    persistent_id = DEVICE_ID_PATH.read_text().strip()

    parts = [
        persistent_id,
        platform.node(),
        platform.machine(),
        str(uuid.getnode()),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _token_hmac_key(fingerprint: str) -> bytes:
    """Device-specific HMAC key for token-cache integrity checking."""
    return hashlib.sha256(fingerprint.encode() + b"|goeckoh-token-v1").digest()


def _load_cached_token() -> Optional[dict]:
    if not TOKEN_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE_PATH.read_text())
        # Verify MAC if present — rejects files edited to extend exp or change plan
        if "mac" in data:
            fp = _device_fingerprint()
            expected = _hmac.new(_token_hmac_key(fp),
                                 data["token"].encode(),
                                 hashlib.sha256).hexdigest()
            if not _hmac.compare_digest(expected, data["mac"]):
                log.warning("Token cache MAC mismatch — possible tampering, discarding")
                TOKEN_CACHE_PATH.unlink(missing_ok=True)
                return None
        return data
    except Exception:
        return None


def _save_token(token: str, plan: str, status: str, warning: Optional[str] = None):
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fp  = _device_fingerprint()
    mac = _hmac.new(_token_hmac_key(fp), token.encode(), hashlib.sha256).hexdigest()
    TOKEN_CACHE_PATH.write_text(json.dumps({
        "token": token,
        "mac": mac,
        "plan": plan,
        "status": status,
        "warning": warning,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }))


def _load_license_key() -> Optional[str]:
    if not LICENSE_KEY_PATH.exists():
        return None
    return LICENSE_KEY_PATH.read_text().strip() or None


def save_license_key(key: str):
    LICENSE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_KEY_PATH.write_text(key.strip().upper())


def _token_valid(token: str) -> bool:
    """Check JWT expiry without verifying signature (server already validated it)."""
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get("exp", 0)
        return time.time() < exp
    except Exception:
        return False


def _token_expires_soon(token: str, within_hours: int = 24) -> bool:
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get("exp", 0)
        return time.time() > (exp - within_hours * 3600)
    except Exception:
        return True


class LicenseClient:
    def __init__(self):
        self._fingerprint = _device_fingerprint()

    def activate(self, license_key: str) -> tuple[LicenseState, str]:
        """
        First-time activation. Call this when user enters their license key.
        Returns (state, message).
        """
        save_license_key(license_key)
        try:
            resp = requests.post(
                f"{LICENSE_SERVER}/license/activate",
                json={
                    "license_key": license_key,
                    "device_fingerprint": self._fingerprint,
                    "platform": f"{platform.system()} {platform.release()}",
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                _save_token(data["token"], data["plan"], data["status"])
                log.info("License activated: %s plan", data["plan"])
                return LicenseState.VALID, f"Activated! Plan: {data['plan'].title()}"
            elif resp.status_code == 403:
                detail = resp.json().get("detail", "")
                if "revoked" in detail.lower():
                    return LicenseState.EXPIRED, "This subscription has lapsed. Please renew at goeckoh.com."
                if "device limit" in detail.lower():
                    return LicenseState.EXPIRED, detail
                return LicenseState.EXPIRED, detail
            elif resp.status_code == 404:
                return LicenseState.NOT_ACTIVATED, "License key not found. Check for typos."
            else:
                return LicenseState.NOT_ACTIVATED, f"Activation error ({resp.status_code})"
        except requests.exceptions.RequestException as e:
            log.warning("Activation network error: %s", e)
            return LicenseState.NOT_ACTIVATED, "Could not reach license server. Check your connection."

    def check(self) -> LicenseState:
        """
        Called on every app startup (and periodically during a session).
        Returns the current license state.

        Priority:
        1. Try to refresh from server (silent, background-friendly)
        2. Fall back to cached token if server unreachable
        3. Report expired if no valid token at all
        """
        license_key = _load_license_key()
        if not license_key:
            return LicenseState.NOT_ACTIVATED

        cached = _load_cached_token()

        # If cached token is valid and not expiring soon, skip server call
        if cached and not _token_expires_soon(cached["token"]):
            status = cached.get("status", "active")
            if status == "grace_period":
                return LicenseState.GRACE_PERIOD
            return LicenseState.OFFLINE_VALID

        # Try server refresh
        try:
            resp = requests.post(
                f"{LICENSE_SERVER}/license/validate",
                json={
                    "license_key": license_key,
                    "device_fingerprint": self._fingerprint,
                },
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                _save_token(data["token"], data["plan"], data["status"], data.get("warning"))
                if data["status"] == "grace_period":
                    return LicenseState.GRACE_PERIOD
                return LicenseState.VALID

            elif resp.status_code == 403:
                detail = resp.json().get("detail", "")
                if "grace_period_expired" in detail or "subscription_lapsed" in detail:
                    # Invalidate cached token so app stops next time even offline
                    if TOKEN_CACHE_PATH.exists():
                        TOKEN_CACHE_PATH.unlink()
                    return LicenseState.EXPIRED
                return LicenseState.EXPIRED

            else:
                log.warning("License server returned %s", resp.status_code)

        except requests.exceptions.RequestException as e:
            log.info("License server unreachable: %s", e)

        # Offline fallback: use cached token if it's still valid
        if cached and _token_valid(cached["token"]):
            status = cached.get("status", "active")
            if status == "grace_period":
                return LicenseState.GRACE_PERIOD
            return LicenseState.OFFLINE_VALID

        # No valid token at all
        return LicenseState.OFFLINE_EXPIRED if cached else LicenseState.NOT_ACTIVATED

    def get_plan(self) -> Optional[str]:
        cached = _load_cached_token()
        if cached:
            return cached.get("plan")
        return None

    def get_warning(self) -> Optional[str]:
        cached = _load_cached_token()
        if cached:
            return cached.get("warning")
        return None


# ---------------------------------------------------------------------------
# Integration: how to wire this into the main Goeckoh pipeline
# ---------------------------------------------------------------------------
#
# In echo_v5_architect.py or the main entrypoint, add:
#
#   from license_client import LicenseClient, LicenseState
#
#   license = LicenseClient()
#   license_state = license.check()
#
#   if license_state == LicenseState.NOT_ACTIVATED:
#       show_activation_screen()
#       sys.exit(0)
#
#   elif license_state in (LicenseState.EXPIRED, LicenseState.OFFLINE_EXPIRED):
#       show_expired_screen()  # "Renew at goeckoh.com"
#       run_read_only_mode()   # data accessible, synthesis disabled
#       sys.exit(0)
#
#   elif license_state == LicenseState.GRACE_PERIOD:
#       show_payment_warning_banner()  # non-blocking, session continues
#
#   # Inject license state into DRCGate so synthesis gating respects it:
#   pipeline = GoeckohPipeline(license_state=license_state)
#
# In DRCGate.evaluate():
#   if self.license_state in (LicenseState.EXPIRED, LicenseState.OFFLINE_EXPIRED):
#       return GateDecision(allow_ai=False, allow_splice=False,
#                           safe_mode=True, reason="license_expired")
#
# ---------------------------------------------------------------------------
