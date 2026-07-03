"""
Creates a live-mode Stripe product + price + payment link.
Run once when ready to take real payments.
Uses the Stripe CLI's own authentication (no key needed in env).
"""
import pathlib, re, subprocess, json

def stripe(*args):
    # --live flag forces live mode; CLI uses its own authenticated session
    r = subprocess.run(['stripe'] + list(args) + ['--live'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('STRIPE ERROR:', r.stdout, r.stderr)
        exit(1)
    return json.loads(r.stdout)

print("Creating live price...")
price = stripe(
    'prices', 'create',
    '--unit-amount', '2000',
    '--currency', 'usd',
    '-d', 'recurring[interval]=month',
    '-d', 'product_data[name]=Goeckoh Monthly',
)
price_id = price['id']
print(f"  price: {price_id}  livemode={price['livemode']}")

print("Creating live payment link...")
link = stripe(
    'payment_links', 'create',
    '-d', f'line_items[0][price]={price_id}',
    '-d', 'line_items[0][quantity]=1',
    '-d', 'after_completion[type]=redirect',
    '-d', 'after_completion[redirect][url]=https://goeckoh.com/download.html?payment=success',
)
url = link['url']
print(f"  payment link: {url}  livemode={link['livemode']}")

# Patch download.html
dl = pathlib.Path(__file__).parent / 'goeckoh-site' / 'download.html'
t = dl.read_text()
t = re.sub(r"const STRIPE_PAYMENT_LINK = '[^']*'", f"const STRIPE_PAYMENT_LINK = '{url}'", t)
dl.write_text(t)
print(f"\n✓ download.html updated with live payment link")
print("  git add goeckoh-site/download.html && git commit -m 'Switch to live Stripe payment link' && git push")
