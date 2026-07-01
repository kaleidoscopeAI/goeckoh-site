#!/usr/bin/env bash
# Run AFTER deploy_fly.sh — registers the production Stripe webhook
# and updates the webhook secret in both .env and fly secrets.

set -e
BACKEND_URL="${1:-https://goeckoh-backend.fly.dev}"
ENV_FILE="$(cd "$(dirname "$0")/backend" && pwd)/.env"
SITE_FILE="$(cd "$(dirname "$0")/goeckoh-site" && pwd)/download.html"

echo "Registering webhook endpoint: $BACKEND_URL/webhook/stripe"

# Create webhook endpoint in Stripe (live mode)
RESULT=$(stripe webhooks create \
  --url "$BACKEND_URL/webhook/stripe" \
  --events checkout.session.completed,customer.subscription.deleted,invoice.payment_failed,invoice.payment_succeeded \
  2>&1)

echo "$RESULT"

# Extract the webhook secret
WHSEC=$(echo "$RESULT" | grep -oP 'whsec_\S+' | head -1)
if [ -z "$WHSEC" ]; then
  echo ""
  echo "Could not extract whsec_ automatically."
  echo "Copy it from the output above and run:"
  echo "  flyctl secrets set STRIPE_WEBHOOK_SECRET=whsec_... -a goeckoh-backend"
  exit 1
fi

echo ""
echo "Webhook secret: ${WHSEC:0:12}... (hidden)"

# Update .env
python3 - <<PYEOF
import re, pathlib
p = pathlib.Path('$ENV_FILE')
t = p.read_text()
t = re.sub(r'^STRIPE_WEBHOOK_SECRET=.*', 'STRIPE_WEBHOOK_SECRET=$WHSEC', t, flags=re.MULTILINE)
p.write_text(t)
print('.env updated')
PYEOF

# Push new secret to Fly
flyctl secrets set "STRIPE_WEBHOOK_SECRET=$WHSEC" -a goeckoh-backend
echo "Fly secret updated."

# Update BACKEND_URL in download.html
python3 - <<PYEOF
import re, pathlib
p = pathlib.Path('$SITE_FILE')
t = p.read_text()
t = re.sub(r"const BACKEND_URL\s*=\s*'[^']*'", "const BACKEND_URL = '$BACKEND_URL'", t)
p.write_text(t)
print('download.html BACKEND_URL updated to $BACKEND_URL')
PYEOF

echo ""
echo "══════════════════════════════════════════════════════"
echo "  COMPLETE — backend fully wired:"
echo "  · Webhook registered at $BACKEND_URL/webhook/stripe"
echo "  · STRIPE_WEBHOOK_SECRET updated in .env + Fly"
echo "  · download.html BACKEND_URL = $BACKEND_URL"
echo ""
echo "  Now commit download.html:"
echo "  cd goeckoh-site && git add download.html && git commit -m 'Set production BACKEND_URL'"
echo "  git push"
echo "══════════════════════════════════════════════════════"
