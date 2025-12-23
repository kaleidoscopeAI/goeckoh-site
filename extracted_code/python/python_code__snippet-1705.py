# First, check whether the default locations from OpenSSL
# seem like they will give us a usable set of CA certs.
# ssl.get_default_verify_paths already takes care of:
# - getting cafile from either the SSL_CERT_FILE env var
#   or the path configured when OpenSSL was compiled,
#   and verifying that that path exists
# - getting capath from either the SSL_CERT_DIR env var
#   or the path configured when OpenSSL was compiled,
#   and verifying that that path exists
# In addition we'll check whether capath appears to contain certs.
defaults = ssl.get_default_verify_paths()
if defaults.cafile or (defaults.capath and _capath_contains_certs(defaults.capath)):
    ctx.set_default_verify_paths()
else:
    # cafile from OpenSSL doesn't exist
    # and capath from OpenSSL doesn't contain certs.
    # Let's search other common locations instead.
    for cafile in _CA_FILE_CANDIDATES:
        if os.path.isfile(cafile):
            ctx.load_verify_locations(cafile=cafile)
            break

yield


