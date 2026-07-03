import re, pathlib, getpass
p = pathlib.Path('/home/jacob/Desktop/goeckoh-platform/backend/.env')
w = getpass.getpass('Paste whsec_ secret (hidden): ')
w = w.strip()
if not w.startswith('whsec_'):
    print('ERROR: expected whsec_... value')
    exit(1)
t = p.read_text()
if re.search(r'^STRIPE_WEBHOOK_SECRET=', t, re.MULTILINE):
    t = re.sub(r'^STRIPE_WEBHOOK_SECRET=.*', f'STRIPE_WEBHOOK_SECRET={w}', t, flags=re.MULTILINE)
else:
    t = t.rstrip() + f'\nSTRIPE_WEBHOOK_SECRET={w}\n'
p.write_text(t)
print('done')
