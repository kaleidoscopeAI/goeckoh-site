#!/usr/bin/env bash
# Goeckoh backend → Fly.io
# Run once to deploy; re-run to redeploy after changes.

set -e
BACKEND_DIR="$(cd "$(dirname "$0")/backend" && pwd)"
ENV_FILE="$BACKEND_DIR/.env"

# ── 1. Install flyctl if needed ─────────────────────────────────────────────
if ! command -v flyctl &>/dev/null; then
  echo "Installing flyctl..."
  curl -L https://fly.io/install.sh | sh
  export PATH="$HOME/.fly/bin:$PATH"
fi

# ── 2. Authenticate (opens browser once) ────────────────────────────────────
flyctl auth whoami 2>/dev/null || flyctl auth login

# ── 3. Create the app (idempotent) ──────────────────────────────────────────
flyctl apps create goeckoh-backend --org personal 2>/dev/null && \
  echo "App created." || echo "App already exists — continuing."

# ── 4. Create persistent volume for SQLite (1 GB, idempotent) ───────────────
flyctl volumes list -a goeckoh-backend 2>/dev/null | grep -q goeckoh_data || \
  flyctl volumes create goeckoh_data --size 1 --region iad -a goeckoh-backend

# ── 5. Push secrets from .env (values never appear in output) ───────────────
echo "Setting secrets..."
grep -E "^(STRIPE_SECRET_KEY|STRIPE_WEBHOOK_SECRET|JWT_SECRET_KEY|ADMIN_SECRET|SMTP_PASS)=" \
  "$ENV_FILE" | flyctl secrets import -a goeckoh-backend
echo "Secrets set."

# ── 6. Build and deploy ──────────────────────────────────────────────────────
cd "$BACKEND_DIR"
flyctl deploy --remote-only -a goeckoh-backend

echo ""
echo "══════════════════════════════════════════════════════"
echo "  DEPLOYED: https://goeckoh-backend.fly.dev"
echo ""
echo "  Next — register Stripe production webhook:"
echo "  stripe webhooks create --url https://goeckoh-backend.fly.dev/webhook/stripe \\"
echo "    --events checkout.session.completed,customer.subscription.deleted,invoice.payment_failed"
echo ""
echo "  Then update BACKEND_URL in goeckoh-site/download.html"
echo "  and set the new whsec_ as a fly secret:"
echo "  flyctl secrets set STRIPE_WEBHOOK_SECRET=whsec_... -a goeckoh-backend"
echo "══════════════════════════════════════════════════════"
