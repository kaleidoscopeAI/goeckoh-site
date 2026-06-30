"""
Goeckoh License & Download Server
----------------------------------
Handles:
  POST /webhook/stripe           — Stripe subscription lifecycle events
  POST /license/activate         — First-time key activation (device registers)
  POST /license/validate         — Refresh JWT token (called on every app startup)
  GET  /download/{platform}      — Gated binary download (requires active license key)
  GET  /health                   — Uptime check

Security posture:
  - Stripe webhooks verified via HMAC signature
  - License keys never sent in query params (always POST body)
  - JWTs signed with HS256, 7-day expiry
  - Device fingerprint binding (max 2 devices per starter license)
  - All secrets from environment variables — never hardcoded
"""

import os
import json
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
# Load environment from .env (backend dir first, then project root) BEFORE anything reads it
for _envpath in (Path(__file__).resolve().parent / ".env",
                 Path(__file__).resolve().parent.parent / ".env"):
    if _envpath.exists():
        load_dotenv(_envpath)
        break

import jwt
import stripe
from fastapi import FastAPI, Request, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import (
    init_db, get_db, generate_license_key,
    License, Device, StripeEvent,
    LicenseStatus, PlanTier,
    User, GuardianLink, UserRole,
)

# ---------------------------------------------------------------------------
# Configuration from environment — all secrets here, nothing hardcoded
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 1

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./downloads"))
SEND_EMAIL = os.environ.get("SEND_EMAIL_ENABLED", "false").lower() == "true"

# Email configuration (optional — any SMTP provider)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "care@goeckoh.com")

# Plan → max devices mapping
PLAN_MAX_DEVICES = {
    PlanTier.STARTER: 2,
    PlanTier.FAMILY: 5,
    PlanTier.CLINICIAN: 10,
}

if not STRIPE_SECRET_KEY:
    logging.warning("STRIPE_SECRET_KEY not set — Stripe operations will fail")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET_KEY must be set in environment")

stripe.api_key = STRIPE_SECRET_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("goeckoh.license")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Goeckoh License Server", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — the browser frontend (goeckoh.com) calls /license/* and /download/* directly.
# Origins are configurable via ALLOWED_ORIGINS (comma-separated) in the environment.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://goeckoh.com,https://www.goeckoh.com,http://localhost:3001",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

init_db()

# Go-live readiness summary — logs one line on boot (and the blockers, if any)
# so a half-configured server is obvious in the logs instead of failing silently
# later (e.g. payment succeeds but no license email goes out). Never fatal; run
# `python preflight.py` for the full report. Never prints secret values.
try:
    from preflight import run_checks
    _pf = run_checks()
    log.info("Preflight: %s", _pf.summary_line())
    for _c in _pf.blockers:
        log.warning("Preflight BLOCKER — %s: %s", _c.name, _c.detail)
except Exception as _e:  # preflight must never stop the server from booting
    log.warning("Preflight check skipped: %s", _e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _issue_jwt(license_key: str, plan: str, email: str) -> str:
    payload = {
        "sub": license_key,
        "plan": plan,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS),
        "iss": "goeckoh.com",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _get_or_create_license_for_subscription(
    db: Session,
    subscription_id: str,
    customer_id: str,
    customer_email: str,
    plan_name: str,
) -> License:
    lic = db.query(License).filter(License.stripe_subscription_id == subscription_id).first()
    if lic:
        return lic

    tier = PlanTier(plan_name) if plan_name in PlanTier._value2member_map_ else PlanTier.STARTER
    lic = License(
        license_key=generate_license_key(),
        customer_email=customer_email,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
        plan=tier,
        status=LicenseStatus.ACTIVE,
        activated_at=datetime.utcnow(),
        max_devices=PLAN_MAX_DEVICES[tier],
    )
    db.add(lic)
    db.commit()
    db.refresh(lic)
    log.info("New license created: %s for %s", lic.license_key, customer_email)
    return lic


def _send_license_email(email: str, license_key: str, plan: str):
    if not SEND_EMAIL:
        log.info("Email disabled — would send license key %s to %s", license_key, email)
        return
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Your Goeckoh License Key"
    msg["From"] = FROM_EMAIL
    msg["To"] = email
    msg.set_content(f"""
Welcome to Goeckoh.

Your license key is:

    {license_key}

To activate:
1. Open the Goeckoh app
2. Go to Settings → License
3. Enter the key above

Your plan: {plan.title()}
Questions: {FROM_EMAIL}

— The Goeckoh Team
""")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)
    log.info("License email sent to %s", email)


# ---------------------------------------------------------------------------
# Stripe Webhook
# ---------------------------------------------------------------------------

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        log.warning("Stripe webhook signature verification failed")
        raise HTTPException(400, "Invalid signature")

    event_id = event["id"]
    event_type = event["type"]

    # Idempotency — skip already-processed events
    existing = db.query(StripeEvent).filter(StripeEvent.id == event_id).first()
    if existing and existing.processed:
        return {"status": "already_processed"}

    if not existing:
        record = StripeEvent(
            id=event_id,
            event_type=event_type,
            payload=json.dumps(event["data"]),
        )
        db.add(record)
        db.commit()

    data = event["data"]["object"]

    # ------------------------------------------------------------------
    if event_type == "checkout.session.completed":
        sub_id = data.get("subscription")
        customer_id = data.get("customer")
        customer_email = data.get("customer_details", {}).get("email") or data.get("customer_email", "")
        plan_name = "starter"

        # Pull plan name from subscription metadata if available
        if sub_id:
            try:
                sub = stripe.Subscription.retrieve(sub_id)
                plan_name = sub.get("metadata", {}).get("plan_name", "starter")
            except Exception:
                pass

        lic = _get_or_create_license_for_subscription(
            db, sub_id, customer_id, customer_email, plan_name
        )
        _send_license_email(customer_email, lic.license_key, plan_name)

    # ------------------------------------------------------------------
    elif event_type == "invoice.payment_succeeded":
        sub_id = data.get("subscription")
        if sub_id:
            lic = db.query(License).filter(License.stripe_subscription_id == sub_id).first()
            if lic and lic.status != LicenseStatus.ACTIVE:
                lic.status = LicenseStatus.ACTIVE
                lic.grace_started_at = None
                lic.revoked_at = None
                db.commit()
                log.info("License %s reactivated after successful payment", lic.license_key)

    # ------------------------------------------------------------------
    elif event_type == "invoice.payment_failed":
        sub_id = data.get("subscription")
        if sub_id:
            lic = db.query(License).filter(License.stripe_subscription_id == sub_id).first()
            if lic and lic.status == LicenseStatus.ACTIVE:
                lic.status = LicenseStatus.GRACE_PERIOD
                lic.grace_started_at = datetime.utcnow()
                db.commit()
                log.info("License %s entered grace period (payment failed)", lic.license_key)

    # ------------------------------------------------------------------
    elif event_type in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub_id = data.get("id")
        stripe_status = data.get("status", "")
        if sub_id:
            lic = db.query(License).filter(License.stripe_subscription_id == sub_id).first()
            if lic and stripe_status in ("canceled", "unpaid", "past_due"):
                lic.status = LicenseStatus.REVOKED
                lic.revoked_at = datetime.utcnow()
                db.commit()
                log.info("License %s revoked (subscription %s)", lic.license_key, stripe_status)

    # Mark event as processed
    record = db.query(StripeEvent).filter(StripeEvent.id == event_id).first()
    if record:
        record.processed = True
        db.commit()

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# License Activation (first-time device registration)
# ---------------------------------------------------------------------------

class ActivateRequest(BaseModel):
    license_key: str
    device_fingerprint: str
    platform: Optional[str] = None


@app.post("/license/activate")
@limiter.limit("5/minute")
async def activate_license(request: Request, body: ActivateRequest, db: Session = Depends(get_db)):
    key = body.license_key.strip().upper()
    lic = db.query(License).filter(License.license_key == key).first()

    if not lic:
        raise HTTPException(404, "License key not found")

    if lic.status == LicenseStatus.REVOKED:
        raise HTTPException(403, "License has been revoked. Please renew your subscription.")

    if lic.status == LicenseStatus.PENDING:
        raise HTTPException(402, "License not yet activated — payment may still be processing.")

    # Check device count
    active_devices = (
        db.query(Device)
        .filter(Device.license_id == lic.id, Device.is_active == True)
        .all()
    )

    device = next(
        (d for d in active_devices if d.device_fingerprint == body.device_fingerprint), None
    )

    if not device:
        if len(active_devices) >= lic.max_devices:
            raise HTTPException(
                403,
                f"Device limit reached ({lic.max_devices} devices for {lic.plan.value} plan). "
                "Deactivate a device at goeckoh.com/account or upgrade your plan."
            )
        device = Device(
            license_id=lic.id,
            device_fingerprint=body.device_fingerprint,
            platform=body.platform,
        )
        db.add(device)
    else:
        device.last_seen = datetime.utcnow()
        device.platform = body.platform or device.platform

    if not lic.activated_at:
        lic.activated_at = datetime.utcnow()

    db.commit()

    token = _issue_jwt(lic.license_key, lic.plan.value, lic.customer_email)
    log.info("Device %s activated for license %s (%s)", body.device_fingerprint[:8], key, lic.plan.value)

    return {
        "token": token,
        "plan": lic.plan.value,
        "expires_in_days": JWT_EXPIRY_DAYS,
        "status": "activated",
    }


# ---------------------------------------------------------------------------
# License Validation (called on every app startup to refresh JWT)
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    license_key: str
    device_fingerprint: str


@app.post("/license/validate")
@limiter.limit("20/minute")
async def validate_license(request: Request, body: ValidateRequest, db: Session = Depends(get_db)):
    key = body.license_key.strip().upper()
    lic = db.query(License).filter(License.license_key == key).first()

    if not lic:
        raise HTTPException(404, "License key not found")

    if lic.status == LicenseStatus.REVOKED:
        raise HTTPException(403, "subscription_lapsed")

    if lic.status == LicenseStatus.PENDING:
        raise HTTPException(402, "payment_pending")

    # Grace period: allow token refresh for 14 days after payment failure
    if lic.status == LicenseStatus.GRACE_PERIOD:
        if lic.grace_started_at:
            days_in_grace = (datetime.utcnow() - lic.grace_started_at).days
            if days_in_grace > 14:
                lic.status = LicenseStatus.REVOKED
                lic.revoked_at = datetime.utcnow()
                db.commit()
                raise HTTPException(403, "grace_period_expired")
        # Issue a shorter token during grace period
        token = _issue_jwt(lic.license_key, lic.plan.value, lic.customer_email)
        return {
            "token": token,
            "plan": lic.plan.value,
            "expires_in_days": 3,
            "status": "grace_period",
            "warning": "Payment failed. Please update your payment method at goeckoh.com/account.",
        }

    # Enforce device limit — same logic as activate so validate can't bypass it
    device = (
        db.query(Device)
        .filter(Device.license_id == lic.id, Device.device_fingerprint == body.device_fingerprint)
        .first()
    )
    if device:
        device.last_seen = datetime.utcnow()
        db.commit()
    else:
        # New device fingerprint seen on validate — enforce limit before issuing token
        active_count = (
            db.query(Device)
            .filter(Device.license_id == lic.id, Device.is_active == True)
            .count()
        )
        if active_count >= lic.max_devices:
            raise HTTPException(
                403,
                f"Device limit reached ({lic.max_devices} devices for {lic.plan.value} plan). "
                "Deactivate a device at goeckoh.com/account or upgrade your plan."
            )
        device = Device(
            license_id=lic.id,
            device_fingerprint=body.device_fingerprint,
            platform=None,
        )
        db.add(device)
        db.commit()
        log.info("New device registered via validate for license %s", key[:12])

    token = _issue_jwt(lic.license_key, lic.plan.value, lic.customer_email)
    return {
        "token": token,
        "plan": lic.plan.value,
        "expires_in_days": JWT_EXPIRY_DAYS,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Device Deactivation
# ---------------------------------------------------------------------------

class DeactivateRequest(BaseModel):
    license_key: str
    device_fingerprint: str


@app.post("/license/deactivate")
@limiter.limit("10/minute")
async def deactivate_device(request: Request, body: DeactivateRequest, db: Session = Depends(get_db)):
    key = body.license_key.strip().upper()
    lic = db.query(License).filter(License.license_key == key).first()

    if not lic:
        raise HTTPException(404, "License key not found")

    device = (
        db.query(Device)
        .filter(
            Device.license_id == lic.id,
            Device.device_fingerprint == body.device_fingerprint,
            Device.is_active == True,
        )
        .first()
    )
    if not device:
        raise HTTPException(404, "Device not found or already deactivated")

    device.is_active = False
    db.commit()
    log.info("Device %s deactivated for license %s", body.device_fingerprint[:8], key[:12])
    return {"status": "deactivated", "device_fingerprint": body.device_fingerprint[:8] + "..."}


# ---------------------------------------------------------------------------
# Gated Binary Downloads
# ---------------------------------------------------------------------------

PLATFORM_FILES = {
    "mac":     "goeckoh-latest-macos.tar.gz",
    "windows": "goeckoh-latest-windows.zip",
    "linux":   "goeckoh-latest-linux.tar.gz",
    "android": "goeckoh-latest-android.apk",   # not yet built
}


@app.get("/download/{platform}")
async def download_binary(
    platform: str,
    db: Session = Depends(get_db),
    license_key: str = Header(..., alias="X-License-Key"),
):
    if platform not in PLATFORM_FILES:
        raise HTTPException(404, f"Unknown platform '{platform}'. Use: {list(PLATFORM_FILES)}")

    key = license_key.strip().upper()
    lic = db.query(License).filter(License.license_key == key).first()

    if not lic:
        raise HTTPException(403, "Invalid license key")

    if lic.status == LicenseStatus.REVOKED:
        raise HTTPException(403, "License has lapsed. Renew at goeckoh.com")

    if lic.status == LicenseStatus.PENDING:
        raise HTTPException(403, "License not yet active")

    filename = PLATFORM_FILES[platform]
    filepath = DOWNLOADS_DIR / filename

    if not filepath.exists():
        log.error("Download file not found: %s", filepath)
        raise HTTPException(503, "Binary not yet available — check back soon.")

    log.info("Download: %s for license %s (%s)", filename, key[:12], lic.plan.value)
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "goeckoh-license", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Account portal (customer-facing frontend) — served same-origin so the
# browser calls /license/* and /download/* without CORS friction.
# ---------------------------------------------------------------------------

_PORTAL = Path(__file__).resolve().parent / "static" / "portal.html"


@app.get("/", include_in_schema=False)
@app.get("/account", include_in_schema=False)
async def account_portal():
    if not _PORTAL.exists():
        raise HTTPException(404, "Portal not found")
    return FileResponse(str(_PORTAL), media_type="text/html")


# ---------------------------------------------------------------------------
# Admin: list licenses (protect with firewall / internal-only route in prod)
# ---------------------------------------------------------------------------

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")


@app.get("/admin/licenses")
async def list_licenses(
    x_admin_secret: str = Header(default=""),
    db: Session = Depends(get_db),
):
    if not ADMIN_SECRET or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

    licenses = db.query(License).order_by(License.created_at.desc()).limit(100).all()
    return [
        {
            "key": lic.license_key,
            "email": lic.customer_email,
            "plan": lic.plan.value,
            "status": lic.status.value,
            "created": lic.created_at.isoformat(),
            "activated": lic.activated_at.isoformat() if lic.activated_at else None,
        }
        for lic in licenses
    ]


# ---------------------------------------------------------------------------
# Auth — password hashing
# ---------------------------------------------------------------------------

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def _issue_user_jwt(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=30),
        "iss": "goeckoh.com",
        "type": "user",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_user_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")


def _current_user(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    payload = _decode_user_jwt(authorization[7:])
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(401, "User not found")
    user.last_seen = datetime.utcnow()
    db.commit()
    return user


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    role: str = "patient"  # "patient" | "guardian" | "clinician"


@app.post("/auth/register")
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email.lower().strip()).first():
        raise HTTPException(409, "An account with that email already exists")

    role_val = body.role if body.role in UserRole._value2member_map_ else "patient"
    user = User(
        email=body.email.lower().strip(),
        password_hash=_hash_password(body.password),
        name=body.name,
        role=UserRole(role_val),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log.info("New user registered: %s (%s)", user.email, user.role.value)

    token = _issue_user_jwt(user.id, user.email, user.role.value)
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role.value}}


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
@limiter.limit("20/minute")
async def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user or not _verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")

    user.last_seen = datetime.utcnow()
    db.commit()

    token = _issue_user_jwt(user.id, user.email, user.role.value)
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role.value}}


@app.get("/auth/me")
async def me(user: User = Depends(_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role.value}


# ---------------------------------------------------------------------------
# Guardian ↔ Patient linking
# ---------------------------------------------------------------------------

@app.post("/guardian/link/{patient_email}")
@limiter.limit("10/minute")
async def link_patient(
    request: Request,
    patient_email: str,
    db: Session = Depends(get_db),
    user: User = Depends(_current_user),
):
    if user.role not in (UserRole.GUARDIAN, UserRole.CLINICIAN):
        raise HTTPException(403, "Only guardian or clinician accounts can link to patients")

    patient = db.query(User).filter(User.email == patient_email.lower().strip()).first()
    if not patient:
        raise HTTPException(404, "No account found with that email")
    if patient.role != UserRole.PATIENT:
        raise HTTPException(400, "That account is not a patient account")

    existing = db.query(GuardianLink).filter(
        GuardianLink.guardian_id == user.id,
        GuardianLink.patient_id == patient.id,
    ).first()
    if existing:
        return {"status": "already_linked", "patient": patient.name or patient.email}

    link = GuardianLink(guardian_id=user.id, patient_id=patient.id)
    db.add(link)
    db.commit()
    log.info("Guardian %s linked to patient %s", user.email, patient.email)
    return {"status": "linked", "patient": patient.name or patient.email}


@app.get("/guardian/patients")
async def list_patients(
    db: Session = Depends(get_db),
    user: User = Depends(_current_user),
):
    if user.role not in (UserRole.GUARDIAN, UserRole.CLINICIAN):
        raise HTTPException(403, "Guardian or clinician accounts only")

    links = db.query(GuardianLink).filter(GuardianLink.guardian_id == user.id).all()
    patients = []
    for link in links:
        p = db.query(User).filter(User.id == link.patient_id).first()
        if p:
            patients.append({
                "id": p.id,
                "name": p.name or p.email,
                "email": p.email,
                "last_seen": p.last_seen.isoformat() if p.last_seen else None,
            })
    return {"patients": patients}


@app.get("/patient/guardians")
async def list_guardians(
    db: Session = Depends(get_db),
    user: User = Depends(_current_user),
):
    links = db.query(GuardianLink).filter(GuardianLink.patient_id == user.id).all()
    guardians = []
    for link in links:
        g = db.query(User).filter(User.id == link.guardian_id).first()
        if g:
            guardians.append({"id": g.id, "name": g.name or g.email, "role": g.role.value})
    return {"guardians": guardians}


# ---------------------------------------------------------------------------
# Real-time relay — live metrics bridge (user device → guardian device)
# Data is never stored. Server is a pipe only.
# ---------------------------------------------------------------------------

import asyncio
import secrets as _secrets

# In-memory relay table: code → { "broadcaster": WS | None, "monitors": [WS, ...] }
_relay: dict[str, dict] = {}


@app.get("/session/new-code")
async def new_session_code(user: User = Depends(_current_user)):
    code = _secrets.token_hex(3).upper()  # 6 hex chars — e.g. "A3F9C1"
    _relay[code] = {"broadcaster": None, "monitors": []}
    log.info("New relay code %s issued for %s", code, user.email)
    return {"code": code}


@app.websocket("/ws/broadcast/{code}")
async def ws_broadcast(websocket: WebSocket, code: str):
    """User's therapy device connects here and sends live metrics JSON."""
    if code not in _relay:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    _relay[code]["broadcaster"] = websocket
    log.info("Broadcaster connected on relay %s", code)
    try:
        while True:
            data = await websocket.receive_text()
            # Fan out to all guardian monitors — do not store
            dead = []
            for monitor_ws in _relay[code]["monitors"]:
                try:
                    await monitor_ws.send_text(data)
                except Exception:
                    dead.append(monitor_ws)
            for d in dead:
                _relay[code]["monitors"].remove(d)
    except WebSocketDisconnect:
        _relay[code]["broadcaster"] = None
        log.info("Broadcaster disconnected from relay %s", code)
        # Notify monitors that session ended
        for monitor_ws in _relay[code]["monitors"]:
            try:
                await monitor_ws.send_text('{"event":"session_ended"}')
            except Exception:
                pass


@app.websocket("/ws/monitor/{code}")
async def ws_monitor(websocket: WebSocket, code: str):
    """Guardian device connects here to receive live metrics."""
    if code not in _relay:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    _relay[code]["monitors"].append(websocket)
    log.info("Guardian monitor connected on relay %s (total monitors: %d)",
             code, len(_relay[code]["monitors"]))
    try:
        while True:
            # Keep connection alive; guardian doesn't send data
            await asyncio.sleep(30)
            await websocket.send_text('{"event":"ping"}')
    except (WebSocketDisconnect, Exception):
        if websocket in _relay[code]["monitors"]:
            _relay[code]["monitors"].remove(websocket)
        log.info("Guardian monitor disconnected from relay %s", code)


# ---------------------------------------------------------------------------
# Session analytics endpoint
# ---------------------------------------------------------------------------
# Default log path matches SessionLogger default in realtime_loop.py.
# Override via SESSION_LOG_PATH environment variable.
_SESSION_LOG_DEFAULT = Path.home() / ".goeckoh" / "sessions" / "session_log.jsonl"

@app.get("/session/stats")
async def session_stats(request: Request,
                        log_path: Optional[str] = None):
    """Return clinical session analytics as JSON (feeds guardian dashboard).

    Reads the JSONL session log written by realtime_loop.SessionLogger.
    Computes: total events, VSA (current vs baseline week), spontaneity %,
    Cohen's d effect size, median latency, formant scatter for last 200 events.
    Returns graceful empty payload when no data exists.
    """
    path = Path(log_path) if log_path else Path(
        os.getenv("SESSION_LOG_PATH", str(_SESSION_LOG_DEFAULT)))

    try:
        import sys as _sys
        _science_path = Path(__file__).parent.parent.parent / "goeckoh-speech-therapy" / "goeckoh"
        if str(_science_path) not in _sys.path:
            _sys.path.insert(0, str(_science_path))
        from science import compute_stats_json
        stats = compute_stats_json(log_path=str(path))
    except Exception as exc:
        log.warning("science.py stats failed: %s", exc)
        stats = {"status": "error", "detail": str(exc), "total_events": 0}

    return JSONResponse(content=stats)


# ---------------------------------------------------------------------------
# Entry point — production: `uvicorn main:app --host 0.0.0.0 --port 8000`
# behind a reverse proxy (HTTPS). Or just `python main.py` for local/dev.
# HOST/PORT configurable via environment.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
