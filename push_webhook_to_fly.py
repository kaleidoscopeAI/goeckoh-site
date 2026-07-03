import pathlib, re, subprocess, os

env_path = pathlib.Path('/home/jacob/Desktop/goeckoh-platform/backend/.env')
text = env_path.read_text()

m = re.search(r'^STRIPE_WEBHOOK_SECRET=(.+)$', text, re.MULTILINE)
if not m or not m.group(1).startswith('whsec_'):
    print('ERROR: whsec_ not found in .env — run set_webhook_secret.py first')
    exit(1)

val = m.group(1).strip()
print(f'Found secret: {val[:12]}... (hidden)')

fly = pathlib.Path.home() / '.fly/bin/flyctl'
result = subprocess.run(
    [str(fly), 'secrets', 'set', f'STRIPE_WEBHOOK_SECRET={val}', '-a', 'goeckoh-backend'],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
else:
    print('Fly secret updated successfully.')
