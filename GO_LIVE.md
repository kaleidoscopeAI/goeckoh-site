# Goeckoh — Go-Live Runbook

Single source of truth for taking the platform live. Everything **code-side is
done and verified**; what remains needs *your* credentials and binaries.

Run the self-check at any point — it tells you exactly what is left:

```bash
cd goeckoh-platform/backend
source venv/bin/activate
python preflight.py        # exit 0 = ready; exit 1 = blockers listed with fixes
```

The same check logs a one-line summary every time the server boots, so a
half-configured server is obvious in the logs (it never fails silently).

---

## Status (auto-verified 2026-06-20)

| Item | State | Owner |
|------|-------|-------|
| dotenv autoload / CORS / `python main.py` launch | ✅ done & verified | — |
| `JWT_SECRET_KEY` (strong, regenerated) | ✅ done | — |
| `ADMIN_SECRET` (strong, generated) | ✅ done | — |
| Database init | ✅ verified | — |
| Website `dist/` rebuilt fresh | ✅ done (`website-for-goechoh/dist/`) | — |
| `STRIPE_SECRET_KEY` | ⛔ placeholder | **you** |
| `STRIPE_WEBHOOK_SECRET` | ⛔ placeholder | **you** |
| Email / SMTP delivery | ⛔ disabled | **you** |
| Installer binaries in `backend/downloads/` | ⛔ empty | **you** |

The 4 blockers below are the *entire* remaining list. Edit `goeckoh-platform/.env`.

---

## 1. Stripe (live payments)  ⛔

1. dashboard.stripe.com → **Developers → API keys** → copy the **live** secret key.
   ```
   STRIPE_SECRET_KEY=sk_live_xxxxxxxx
   ```
2. **Developers → Webhooks → Add endpoint**
   - URL: `https://api.goeckoh.com/webhook/stripe` (your deployed backend URL)
   - Events: `checkout.session.completed`, `invoice.payment_succeeded`,
     `invoice.payment_failed`, `customer.subscription.deleted`,
     `customer.subscription.updated`
   - Copy the **Signing secret**:
   ```
   STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxx
   ```
3. The checkout uses 3 hardcoded Payment Links in
   `website-for-goechoh/pages/GetApp.tsx` (`buy.stripe.com/...`). Confirm those
   point at your **live** products, and set each product's price metadata
   `plan_name` to `starter` / `family` / `clinician` so the webhook assigns the
   right device limit (otherwise it defaults to `starter` = 2 devices).

## 2. Email delivery (buyers receive their key)  ⛔

Without this, a purchase succeeds but no key is sent (recoverable manually via
`/admin/licenses`, but that is not a launch posture).
```
SEND_EMAIL_ENABLED=true
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USER=postmaster@goeckoh.com
SMTP_PASS=your_smtp_password
FROM_EMAIL=care@goeckoh.com
```
> Keep empty SMTP fields *blank with no inline comment* — python-dotenv would
> otherwise capture the comment text as the value.

## 3. Installer binaries  ⛔

`/download/{platform}` returns 503 until the files exist. Stage them in
`goeckoh-platform/backend/downloads/` (or set `DOWNLOADS_DIR`):
```
goeckoh-latest-mac.dmg
goeckoh-latest-win.exe
goeckoh-latest-linux.deb
goeckoh-latest-android.apk
```
You can launch with a subset — only the missing platforms 503; preflight warns.

---

## 4. Deploy

```bash
cd goeckoh-platform/backend
source venv/bin/activate
python preflight.py            # must be green (or only acceptable warnings)
```
Run behind HTTPS (the JWTs and license keys must never travel in cleartext):
- A reverse proxy (nginx/Caddy) terminating TLS → forwards to `127.0.0.1:8000`.
- A process manager so it restarts on crash/reboot. A unit file already exists:
  `backend/goeckoh-backend.service` — review `WorkingDirectory`/`ExecStart`,
  point `EnvironmentFile` at the `.env`, then `systemctl enable --now`.

Smoke test after deploy:
```bash
curl https://api.goeckoh.com/health                       # {"status":"ok",...}
curl -s https://api.goeckoh.com/download/mac -H 'X-License-Key: x'  # 403 (not 500)
```

## 5. Website

Already rebuilt — deploy `website-for-goechoh/dist/` to your static host
(point `goeckoh.com` at it). Rebuild after any source change:
```bash
cd website-for-goechoh && npm run build
```

---

## Note — duplicate backends

`backend/main.py` is replicated in `license_server/`, `goeckoh/server/`, and
`goeckoh_vc_engine/server/`. **`goeckoh-platform/` is canonical** (it holds the
real `.env` and these fixes). Deploy *only* this one; the others are stale and
should not be pushed to production.
