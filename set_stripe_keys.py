import pathlib, re, getpass

stripe_cfg = pathlib.Path.home() / ".config/stripe/config.toml"
env_path   = pathlib.Path("/home/jacob/Desktop/goeckoh-platform/backend/.env")

cfg = stripe_cfg.read_text()

# Pull keys from stripe CLI config (already on this machine — never shown in output)
test_key = re.search(r"test_mode_api_key = '([^']+)'", cfg).group(1)
live_key = re.search(r"live_mode_api_key = '([^']+)'", cfg).group(1)

print("Which Stripe key do you want to use?")
print("  t = test mode (safe, no real charges)")
print("  l = live mode (real payments)")
choice = input("t or l [t]: ").strip().lower() or "t"
sk = test_key if choice != "l" else live_key

# Webhook secret — paste from `stripe listen` output
print()
print("Now paste your STRIPE_WEBHOOK_SECRET (whsec_...).")
print("Get it by running in another terminal:")
print("  stripe listen --forward-to localhost:8000/webhook/stripe")
print("and copying the 'Your webhook signing secret is whsec_...' line.")
whsec = getpass.getpass("whsec_: ").strip()

t = env_path.read_text()

def upsert(text, key, val):
    if re.search(rf'^{key}=', text, re.MULTILINE):
        return re.sub(rf'^{key}=.*', f'{key}={val}', text, flags=re.MULTILINE)
    return text.rstrip() + f'\n{key}={val}\n'

t = upsert(t, 'STRIPE_SECRET_KEY', sk)
t = upsert(t, 'STRIPE_WEBHOOK_SECRET', whsec)
env_path.write_text(t)

print("done — keys written, nothing printed to screen")
