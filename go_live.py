#!/usr/bin/env python3
"""
go_live.py — One-shot Goeckoh live Stripe setup + deploy
  • reads sk_live_ from backend/.env
  • creates product, price, payment link in live mode
  • registers live webhook on goeckoh-backend.fly.dev
  • writes whsec_ back to .env and pushes to Fly secrets
  • patches download.html with live payment link
  • git commits + pushes everything

Run:  python3 go_live.py
"""
import json, pathlib, re, subprocess, sys

ROOT        = pathlib.Path(__file__).parent
DHTML       = ROOT / 'goeckoh-site' / 'download.html'
ENVF        = ROOT / 'backend' / '.env'
FLY         = str(pathlib.Path.home() / '.fly/bin/flyctl')
FLY_APP     = 'goeckoh-backend'
BACKEND_URL = 'https://goeckoh-backend.fly.dev'
STRIPE_BIN  = 'stripe'

EVENTS = [
    'checkout.session.completed',
    'customer.subscription.created',
    'customer.subscription.updated',
    'customer.subscription.deleted',
    'invoice.payment_succeeded',
    'invoice.payment_failed',
]

# ── helpers ────────────────────────────────────────────────────────────────

def run(cmd, label=''):
    tag = label or ' '.join(str(c) for c in cmd[:3])
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'\n[FAIL] {tag}')
        if r.stdout: print(r.stdout[-3000:])
        if r.stderr: print(r.stderr[-3000:])
        sys.exit(1)
    return r.stdout.strip()

def stripe(*args, key):
    """Call stripe CLI in live mode with explicit secret key."""
    out = run([STRIPE_BIN, '-k', key, '--live'] + list(args), label=f'stripe {args[0]} {args[1]}')
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        print(f'[FAIL] stripe returned non-JSON:\n{out[:1000]}')
        sys.exit(1)

def read_env(path):
    pairs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            pairs[k.strip()] = v.strip()
    return pairs

def write_env_key(path, key, value):
    txt = path.read_text()
    if re.search(rf'^{re.escape(key)}=', txt, re.MULTILINE):
        txt = re.sub(rf'^{re.escape(key)}=.*', f'{key}={value}', txt, flags=re.MULTILINE)
    else:
        txt = txt.rstrip() + f'\n{key}={value}\n'
    path.write_text(txt)

# ── main ───────────────────────────────────────────────────────────────────

def main():
    print('╔══════════════════════════════════════╗')
    print('║   Goeckoh — Go Live (Stripe + Fly)  ║')
    print('╚══════════════════════════════════════╝\n')

    # 1. Read sk_live_ from .env
    env = read_env(ENVF)
    sk = env.get('STRIPE_SECRET_KEY', '')
    if not sk.startswith('sk_live_'):
        print('ERROR: STRIPE_SECRET_KEY in backend/.env must be sk_live_...')
        print(f'       Found: {sk[:12]}...')
        print('       Add your sk_live_ key to backend/.env as STRIPE_SECRET_KEY=sk_live_...')
        sys.exit(1)
    print(f'[1/8] sk_live_ found in .env  ({sk[:16]}...)\n')

    # 2. Verify stripe CLI
    try:
        v = run([STRIPE_BIN, '--version'], label='stripe version')
        print(f'[2/8] Stripe CLI: {v}\n')
    except FileNotFoundError:
        print('ERROR: stripe CLI not found. Install: https://stripe.com/docs/stripe-cli')
        sys.exit(1)

    # 3. Create product
    print('[3/8] Creating live product...')
    product = stripe('products', 'create',
        '-d', 'name=Goeckoh Monthly',
        '-d', 'description=Real-time voice correction for neurodiverse speakers',
        key=sk,
    )
    prod_id = product['id']
    print(f'      product id : {prod_id}\n')

    # 4. Create price
    print('[4/8] Creating live price — $20/month...')
    price = stripe('prices', 'create',
        '-d', f'product={prod_id}',
        '-d', 'unit_amount=2000',
        '-d', 'currency=usd',
        '-d', 'recurring[interval]=month',
        key=sk,
    )
    price_id = price['id']
    print(f'      price id   : {price_id}\n')

    # 5. Create payment link
    print('[5/8] Creating live payment link...')
    link = stripe('payment_links', 'create',
        '-d', f'line_items[0][price]={price_id}',
        '-d', 'line_items[0][quantity]=1',
        '-d', 'after_completion[type]=redirect',
        '-d', 'after_completion[redirect][url]=https://goeckoh.com/download.html?payment=success',
        key=sk,
    )
    pay_url = link['url']
    print(f'      payment link: {pay_url}\n')

    # 6. Register live webhook
    print(f'[6/8] Registering live webhook → {BACKEND_URL}/stripe/webhook ...')
    event_args = []
    for i, ev in enumerate(EVENTS):
        event_args += ['-d', f'enabled_events[{i}]={ev}']
    webhook = stripe('webhook_endpoints', 'create',
        '-d', f'url={BACKEND_URL}/stripe/webhook',
        *event_args,
        key=sk,
    )
    whsec  = webhook['secret']
    wh_id  = webhook['id']
    print(f'      webhook id : {wh_id}')
    print(f'      whsec_     : (hidden)\n')

    # 7. Write whsec_ to backend/.env
    print('[7/8] Updating backend/.env and Fly secrets...')
    write_env_key(ENVF, 'STRIPE_WEBHOOK_SECRET', whsec)
    print('      .env updated')
    run([FLY, 'secrets', 'set', f'STRIPE_WEBHOOK_SECRET={whsec}', '-a', FLY_APP],
        label='flyctl secrets set STRIPE_WEBHOOK_SECRET')
    print('      Fly secret pushed\n')

    # 8. Patch download.html + git commit + push
    print('[8/8] Patching download.html and deploying...')
    html = DHTML.read_text()
    html = re.sub(
        r"const STRIPE_PAYMENT_LINK\s*=\s*'[^']*'",
        f"const STRIPE_PAYMENT_LINK = '{pay_url}'",
        html,
    )
    DHTML.write_text(html)
    print('      download.html updated')

    run(['git', 'add', str(DHTML)], label='git add')
    run(['git', 'commit', '-m',
         f'Switch to live Stripe payment link\n\nProduct: {prod_id}\nPrice: {price_id}\nWebhook: {wh_id}'],
        label='git commit')
    run(['git', 'push', 'origin', 'main'], label='git push')
    print('      pushed to GitHub Pages\n')

    print('╔══════════════════════════════════════════════════════════╗')
    print('║                    LIVE — ALL DONE                      ║')
    print('╚══════════════════════════════════════════════════════════╝')
    print(f'\n  Payment link : {pay_url}')
    print(f'  Webhook ID   : {wh_id}')
    print(f'  Webhook URL  : {BACKEND_URL}/stripe/webhook')
    print(f'  Live page    : https://goeckoh.com/download.html')


if __name__ == '__main__':
    main()
