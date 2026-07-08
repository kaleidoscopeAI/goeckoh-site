"""
Database schema for the Goeckoh license server.
Uses SQLite for simple single-server deployments.
Swap DATABASE_URL to PostgreSQL for multi-server scaling.
"""

import os
import uuid
import secrets
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    create_engine, Column, String, DateTime, Boolean,
    Text, Integer, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./goeckoh_licenses.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class LicenseStatus(str, Enum):
    ACTIVE = "active"
    GRACE_PERIOD = "grace_period"
    REVOKED = "revoked"
    PENDING = "pending"


class PlanTier(str, Enum):
    STARTER = "starter"
    FAMILY = "family"
    CLINICIAN = "clinician"


class License(Base):
    __tablename__ = "licenses"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    license_key = Column(String(32), unique=True, nullable=False, index=True)
    customer_email = Column(String, nullable=False, index=True)
    stripe_customer_id = Column(String, nullable=True, index=True)
    stripe_subscription_id = Column(String, nullable=True, index=True, unique=True)
    plan = Column(SAEnum(PlanTier), nullable=False, default=PlanTier.STARTER)
    status = Column(SAEnum(LicenseStatus), nullable=False, default=LicenseStatus.PENDING)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)
    grace_started_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    max_devices = Column(Integer, nullable=False, default=2)
    notes = Column(Text, nullable=True)


class PromoCode(Base):
    """
    Shared promo/comp codes (press, conferences, partner clinics) — each
    redemption creates a real License row via /promo/redeem, just without a
    Stripe subscription behind it. max_redemptions caps how many people can
    use the same code; redemption_count tracks usage so far.
    """
    __tablename__ = "promo_codes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(64), unique=True, nullable=False, index=True)
    plan = Column(SAEnum(PlanTier), nullable=False, default=PlanTier.STARTER)
    max_redemptions = Column(Integer, nullable=False, default=1)
    redemption_count = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class Device(Base):
    __tablename__ = "devices"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    license_id = Column(String, nullable=False, index=True)
    device_fingerprint = Column(String, nullable=False)
    platform = Column(String, nullable=True)
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)


class StripeEvent(Base):
    __tablename__ = "stripe_events"

    id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False)
    received_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed = Column(Boolean, nullable=False, default=False)
    payload = Column(Text, nullable=True)


class UserRole(str, Enum):
    PATIENT = "patient"
    GUARDIAN = "guardian"
    CLINICIAN = "clinician"


class User(Base):
    """
    Account identity only — no voice data, no session metrics, no PHI.
    All therapeutic data lives in IndexedDB on the user's device.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String, nullable=True)
    role = Column(SAEnum(UserRole), nullable=False, default=UserRole.PATIENT)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=True)


class GuardianLink(Base):
    """
    Which guardian accounts are linked to which patient accounts.
    No health data — just the relationship.
    """
    __tablename__ = "guardian_links"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    guardian_id = Column(String, nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_license_key() -> str:
    """Generates a human-friendly license key: GOEK-XXXX-XXXX-XXXX"""
    raw = secrets.token_hex(6).upper()
    return f"GOEK-{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"
