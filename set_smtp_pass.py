import pathlib, re, getpass
p = pathlib.Path('/home/jacob/Desktop/goeckoh-platform/backend/.env')
pw = getpass.getpass('Gmail app password: ')
t = p.read_text()
if 'SMTP_PASS=' not in t:
    t = t.rstrip() + f'\nSMTP_PASS={pw}\n'
else:
    t = re.sub(r'^SMTP_PASS=.*', f'SMTP_PASS={pw}', t, flags=re.MULTILINE)
p.write_text(t)
print('done')
