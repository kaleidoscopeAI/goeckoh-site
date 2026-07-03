#!/usr/bin/env python3
"""
Goeckoh secret setter — run this to add Stripe and SMTP credentials to backend/.env
Keys are entered via getpass (never visible in terminal, never in chat history).
"""
import getpass
import os
import re
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / "backend" / ".env"

def read_env(path: Path) -> dict:
    pairs = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                pairs[k.strip()] = v.strip()
    return pairs

def write_env(path: Path, pairs: dict):
    lines = []
    if path.exists():
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                k = stripped.split('=', 1)[0].strip()
                if k in pairs:
                    lines.append(f"{k}={pairs.pop(k)}")
                    continue
            lines.append(line)
    # Append any new keys that weren't already in the file
    for k, v in pairs.items():
        lines.append(f"{k}={v}")
    path.write_text('\n'.join(lines) + '\n')

def prompt_secret(label: str, current_redacted: str = '') -> str:
    hint = f" [{current_redacted}]" if current_redacted else " [not set]"
    print(f"\n{label}{hint}")
    print("  Press Enter to keep existing value, or paste/type new value:")
    val = getpass.getpass("  > ")
    return val.strip()

def main():
    print("=" * 60)
    print("  GOECKOH SECRET SETTER")
    print("  Keys entered here are written to backend/.env")
    print("  They are NOT logged, NOT visible in terminal history.")
    print("=" * 60)

    current = read_env(ENV_FILE)
    updates = {}

    def redact(key):
        v = current.get(key, '')
        if not v or len(v) < 8:
            return ''
        return v[:6] + '...' + v[-4:]

    # ── Stripe ──────────────────────────────────────────────────────────────
    print("\n── STRIPE ─────────────────────────────────────────────────────")
    print("  dashboard.stripe.com → Developers → API keys → live secret key")

    sk = prompt_secret("STRIPE_SECRET_KEY (sk_live_...)", redact('STRIPE_SECRET_KEY'))
    if sk:
        if not sk.startswith('sk_'):
            print("  ⚠  That doesn't look like a Stripe secret key (should start with sk_). Skipping.")
        else:
            updates['STRIPE_SECRET_KEY'] = sk

    print("\n  dashboard.stripe.com → Developers → Webhooks → your endpoint → Signing secret")
    wh = prompt_secret("STRIPE_WEBHOOK_SECRET (whsec_...)", redact('STRIPE_WEBHOOK_SECRET'))
    if wh:
        if not wh.startswith('whsec_'):
            print("  ⚠  That doesn't look like a webhook secret (should start with whsec_). Skipping.")
        else:
            updates['STRIPE_WEBHOOK_SECRET'] = wh

    # ── SMTP ────────────────────────────────────────────────────────────────
    print("\n── EMAIL DELIVERY ──────────────────────────────────────────────")
    print("  Buyers receive their license key by email after payment.")
    setup_smtp = input("\n  Set up email now? [y/N] ").strip().lower()
    if setup_smtp == 'y':
        host = input("  SMTP_HOST (e.g. smtp.gmail.com): ").strip()
        if host:
            updates['SMTP_HOST'] = host
        port = input("  SMTP_PORT [587]: ").strip()
        updates['SMTP_PORT'] = port if port else '587'
        user = input("  SMTP_USER (your email address or username): ").strip()
        if user:
            updates['SMTP_USER'] = user
        smtp_pass = prompt_secret("SMTP_PASS")
        if smtp_pass:
            updates['SMTP_PASS'] = smtp_pass
        from_addr = input(f"  FROM_EMAIL [{current.get('FROM_EMAIL','care@goeckoh.com')}]: ").strip()
        updates['FROM_EMAIL'] = from_addr if from_addr else current.get('FROM_EMAIL', 'care@goeckoh.com')
        updates['SEND_EMAIL_ENABLED'] = 'true'
        print("  ✓ Email delivery enabled")

    if not updates:
        print("\nNo changes made.")
        return

    write_env(ENV_FILE, updates)
    print(f"\n✓ {len(updates)} key(s) written to {ENV_FILE}")
    print("\nRun preflight to check readiness:")
    print("  cd backend && source venv/bin/activate && python3 preflight.py")

if __name__ == '__main__':
    main()
