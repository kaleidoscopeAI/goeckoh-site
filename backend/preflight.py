#!/usr/bin/env python3
"""
Goeckoh go-live preflight check.

Verifies that everything required to take real money and deliver licenses is
configured BEFORE the server is exposed to the public. Run it on the box you
intend to deploy on:

    python preflight.py            # human-readable report; exit 1 if any BLOCKER
    python preflight.py --json     # machine-readable (for monitoring / CI)

It is also imported by main.py to log a one-line readiness summary on startup,
so a half-configured server announces itself in the logs instead of failing
silently at the worst moment (e.g. payment succeeds but no license email goes
out because SMTP was never set).

No secret VALUES are ever printed — only whether each one is present and
plausible. Safe to run anywhere, safe to paste the output.

Severity:
  BLOCKER  — go-live is unsafe/broken until fixed (placeholder keys, no binaries…)
  WARNING  — works, but you probably don't want to launch like this
  OK       — configured
"""
from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env exactly like main.py does, so preflight sees the same config the
# server will: backend dir first, then project root.
try:
    from dotenv import load_dotenv
    for _envpath in (Path(__file__).resolve().parent / ".env",
                     Path(__file__).resolve().parent.parent / ".env"):
        if _envpath.exists():
            load_dotenv(_envpath)
            break
except ImportError:
    pass  # main.py will already have loaded it; standalone without dotenv = env only

BLOCKER, WARNING, OK = "BLOCKER", "WARNING", "OK"

# Values that mean "the developer never filled this in".
_PLACEHOLDERS = ("placeholder", "replace_me", "dev_key", "changeme", "todo", "xxx")

# The fixed dev JWT secret that shipped in the repo — must not survive to prod.
_DEV_JWT = "a7d2e9f1b4c6"

PLATFORM_FILES = {
    "mac":     "goeckoh-latest-macos.tar.gz",
    "windows": "goeckoh-latest-windows.zip",
    "linux":   "goeckoh-latest-linux.tar.gz",
    "android": "goeckoh-latest-android.apk",
}


def _is_placeholder(v: str) -> bool:
    lv = v.strip().lower()
    return (not lv) or any(p in lv for p in _PLACEHOLDERS)


@dataclass
class Check:
    name: str
    level: str          # BLOCKER / WARNING / OK
    detail: str         # what we found (never a secret value)
    fix: str = ""       # how to resolve it


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, *a, **k):
        self.checks.append(Check(*a, **k))

    @property
    def blockers(self):  return [c for c in self.checks if c.level == BLOCKER]
    @property
    def warnings(self):  return [c for c in self.checks if c.level == WARNING]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def summary_line(self) -> str:
        if self.ready and not self.warnings:
            return "GO-LIVE READY ✓  all checks pass"
        return (f"NOT go-live ready — {len(self.blockers)} blocker(s), "
                f"{len(self.warnings)} warning(s). Run `python preflight.py`.")


def run_checks() -> Report:
    r = Report()

    # ── JWT signing secret ───────────────────────────────────────────────────
    jwt = os.environ.get("JWT_SECRET_KEY", "")
    if not jwt:
        r.add("JWT_SECRET_KEY", BLOCKER, "not set",
              'python -c "import secrets; print(secrets.token_hex(32))" → put in .env')
    elif _DEV_JWT in jwt or len(jwt) < 32:
        r.add("JWT_SECRET_KEY", BLOCKER, "still the shipped dev value or too short",
              'regenerate: python -c "import secrets; print(secrets.token_hex(32))"')
    else:
        r.add("JWT_SECRET_KEY", OK, f"set ({len(jwt)} chars)")

    # ── Stripe live keys ─────────────────────────────────────────────────────
    sk = os.environ.get("STRIPE_SECRET_KEY", "")
    if _is_placeholder(sk):
        r.add("STRIPE_SECRET_KEY", BLOCKER, "placeholder / unset — no payments can be taken",
              "set sk_live_... from dashboard.stripe.com → Developers → API keys")
    elif sk.startswith("sk_test_"):
        r.add("STRIPE_SECRET_KEY", WARNING, "TEST key (sk_test_) — sandbox only, not real sales",
              "swap to sk_live_... before public launch")
    elif sk.startswith("sk_live_") or sk.startswith("rk_live_"):
        # This app calls Checkout Session and Subscription reads (verify-session,
        # webhook plan lookup). A restricted key (rk_live_) that's the wrong one
        # or missing those scopes authenticates fine but 403s on exactly those
        # calls — the prefix alone doesn't prove it works, so actually call it.
        try:
            import stripe as _stripe
            _stripe.api_key = sk
            _stripe.checkout.Session.list(limit=1)
            _stripe.Subscription.list(limit=1)
            kind = "restricted" if sk.startswith("rk_live_") else "full secret"
            r.add("STRIPE_SECRET_KEY", OK,
                  f"live {kind} key present and can read Checkout Sessions + Subscriptions")
        except Exception as e:
            r.add("STRIPE_SECRET_KEY", BLOCKER,
                  f"live key present but the API rejected a real call: {e}",
                  "if this is a restricted key (rk_live_), grant it Checkout Sessions: Read "
                  "and Subscriptions: Read, or switch to the full sk_live_ secret key")
    else:
        r.add("STRIPE_SECRET_KEY", WARNING, "set but not a recognised Stripe key format",
              "expected sk_live_... or rk_live_... (or sk_test_... for sandbox)")

    whsec = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if _is_placeholder(whsec):
        r.add("STRIPE_WEBHOOK_SECRET", BLOCKER,
              "placeholder / unset — webhooks rejected, licenses never created",
              "create the webhook endpoint in Stripe → copy the whsec_... signing secret")
    elif not whsec.startswith("whsec_"):
        r.add("STRIPE_WEBHOOK_SECRET", WARNING, "set but not in whsec_... format",
              "verify it is the endpoint's signing secret")
    else:
        r.add("STRIPE_WEBHOOK_SECRET", OK, "set (whsec_)")

    # A correct whsec_ doesn't prove an endpoint pointing at THIS deployment is
    # actually registered in Stripe — that's the failure mode that leaves every
    # real payment stuck (session verified, but no License row, no email, ever).
    if not _is_placeholder(sk):
        try:
            import stripe as _stripe
            _stripe.api_key = sk
            endpoints = _stripe.WebhookEndpoint.list(limit=20)
            live_matches = [
                e for e in endpoints.auto_paging_iter()
                if e.get("url", "").rstrip("/").endswith("/webhook/stripe")
                and e.get("status") == "enabled"
            ]
            if not live_matches:
                r.add("Stripe webhook endpoint", BLOCKER,
                      "no enabled endpoint in this Stripe account targets */webhook/stripe — "
                      "checkout.session.completed will never reach this server",
                      "dashboard.stripe.com → Developers → Webhooks → Add endpoint → "
                      "https://<this-host>/webhook/stripe, events: checkout.session.completed, "
                      "invoice.payment_succeeded, invoice.payment_failed, "
                      "customer.subscription.updated, customer.subscription.deleted")
            else:
                needed = {"checkout.session.completed"}
                for e in live_matches:
                    events = set(e.get("enabled_events", []))
                    if "*" in events or needed <= events:
                        r.add("Stripe webhook endpoint", OK,
                              f"{e['url']} is enabled and subscribed to checkout.session.completed")
                        break
                else:
                    r.add("Stripe webhook endpoint", BLOCKER,
                          f"endpoint {live_matches[0]['url']} exists but isn't subscribed to "
                          "checkout.session.completed — licenses will never be created",
                          "edit the endpoint in the Stripe dashboard and add that event")
        except Exception as e:
            r.add("Stripe webhook endpoint", WARNING,
                  f"could not list webhook endpoints to verify ({e})",
                  "this key may lack Webhook Endpoints: Read — check manually in the dashboard")

    # ── Email delivery (license keys reach buyers) ───────────────────────────
    send = os.environ.get("SEND_EMAIL_ENABLED", "false").lower() == "true"
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    if not send:
        r.add("Email delivery", BLOCKER,
              "SEND_EMAIL_ENABLED=false — buyers will NOT receive their license key",
              "fill SMTP_* and set SEND_EMAIL_ENABLED=true (keys still recoverable via /admin/licenses)")
    elif not (smtp_host and smtp_user and smtp_pass):
        r.add("Email delivery", BLOCKER,
              "enabled but SMTP_HOST/USER/PASS incomplete — send will crash the webhook",
              "set SMTP_HOST, SMTP_USER, SMTP_PASS")
    else:
        r.add("Email delivery", OK, f"enabled via {smtp_host}")

    # ── Installer binaries (served from a private GitHub release, not disk) ──
    gh_repo = os.environ.get("GITHUB_RELEASES_REPO", "kaleidoscopeAI/goeckoh-releases")
    gh_token = os.environ.get("GITHUB_RELEASES_TOKEN", "")
    if _is_placeholder(gh_token):
        r.add("Installer binaries", BLOCKER,
              "GITHUB_RELEASES_TOKEN not set — every /download returns 503",
              "create a fine-grained PAT with Contents:read on the releases repo, "
              "set GITHUB_RELEASES_TOKEN")
    else:
        try:
            import requests
            resp = requests.get(
                f"https://api.github.com/repos/{gh_repo}/releases/latest",
                headers={"Authorization": f"Bearer {gh_token}",
                         "Accept": "application/vnd.github+json"},
                timeout=10,
            )
            if resp.status_code != 200:
                r.add("Installer binaries", BLOCKER,
                      f"GitHub releases lookup failed ({resp.status_code}) for {gh_repo} — "
                      "every /download returns 503",
                      "check GITHUB_RELEASES_TOKEN has Contents:read on that repo, and that "
                      "GITHUB_RELEASES_REPO is correct")
            else:
                names = {a["name"] for a in resp.json().get("assets", [])}
                present = [p for p, f in PLATFORM_FILES.items() if f in names]
                missing = [p for p in PLATFORM_FILES if p not in present]
                if not present:
                    r.add("Installer binaries", BLOCKER,
                          f"latest release in {gh_repo} has none of the expected assets",
                          f"upload: {', '.join(PLATFORM_FILES.values())}")
                elif missing:
                    r.add("Installer binaries", WARNING,
                          f"present: {', '.join(present)}; missing: {', '.join(missing)}",
                          "those platforms' downloads will 503 until uploaded")
                else:
                    r.add("Installer binaries", OK,
                          f"all four platforms found in {gh_repo} latest release")
        except Exception as e:
            r.add("Installer binaries", WARNING,
                  f"could not reach GitHub to verify ({e})",
                  "check network egress to api.github.com from this host")

    # ── Admin endpoint guard ─────────────────────────────────────────────────
    admin = os.environ.get("ADMIN_SECRET", "")
    if not admin:
        r.add("ADMIN_SECRET", WARNING,
              "empty — /admin/licenses refuses ALL requests (safe, but you can't view licenses)",
              'set a strong value: python -c "import secrets; print(secrets.token_urlsafe(32))"')
    elif len(admin) < 16:
        r.add("ADMIN_SECRET", WARNING, "set but short / guessable",
              "use a 32-byte random value")
    else:
        r.add("ADMIN_SECRET", OK, "set")

    # ── CORS origins ─────────────────────────────────────────────────────────
    origins = os.environ.get("ALLOWED_ORIGINS", "")
    if "localhost" in origins and "https://" not in origins:
        r.add("ALLOWED_ORIGINS", WARNING, "only localhost origins — the live site can't call the API",
              "include https://goeckoh.com (and www)")
    elif not origins:
        r.add("ALLOWED_ORIGINS", WARNING, "unset — falling back to built-in defaults", "")
    else:
        r.add("ALLOWED_ORIGINS", OK, origins)

    # ── Database reachable / writable ────────────────────────────────────────
    try:
        from models import init_db  # noqa
        init_db()
        r.add("Database", OK, "init_db() succeeded")
    except Exception as e:  # pragma: no cover - environment dependent
        r.add("Database", BLOCKER, f"init_db() failed: {e}",
              "check DB path / DATABASE_URL and write permissions")

    return r


def _print_report(r: Report) -> None:
    icon = {BLOCKER: "✗", WARNING: "!", OK: "✓"}
    width = max(len(c.name) for c in r.checks)
    print("\n  GOECKOH GO-LIVE PREFLIGHT")
    print("  " + "=" * 60)
    for c in r.checks:
        print(f"  [{icon[c.level]}] {c.name.ljust(width)}  {c.detail}")
        if c.fix and c.level != OK:
            print(f"        ↳ fix: {c.fix}")
    print("  " + "=" * 60)
    print(f"  {r.summary_line()}\n")


def main() -> int:
    r = run_checks()
    if "--json" in sys.argv:
        print(json.dumps({
            "ready": r.ready,
            "summary": r.summary_line(),
            "checks": [c.__dict__ for c in r.checks],
        }, indent=2))
    else:
        _print_report(r)
    return 0 if r.ready else 1


if __name__ == "__main__":
    sys.exit(main())
