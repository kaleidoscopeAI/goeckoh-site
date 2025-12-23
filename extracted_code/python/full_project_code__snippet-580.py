"""
Load certificates and maybe keys from a number of files. Has the end goal
of returning a CFArray containing one SecIdentityRef, and then zero or more
SecCertificateRef objects, suitable for use as a client certificate trust
chain.
"""
# Ok, the strategy.
#
# This relies on knowing that macOS will not give you a SecIdentityRef
# unless you have imported a key into a keychain. This is a somewhat
# artificial limitation of macOS (for example, it doesn't necessarily
# affect iOS), but there is nothing inside Security.framework that lets you
# get a SecIdentityRef without having a key in a keychain.
#
# So the policy here is we take all the files and iterate them in order.
# Each one will use SecItemImport to have one or more objects loaded from
# it. We will also point at a keychain that macOS can use to work with the
# private key.
#
# Once we have all the objects, we'll check what we actually have. If we
# already have a SecIdentityRef in hand, fab: we'll use that. Otherwise,
# we'll take the first certificate (which we assume to be our leaf) and
# ask the keychain to give us a SecIdentityRef with that cert's associated
# key.
#
# We'll then return a CFArray containing the trust chain: one
# SecIdentityRef and then zero-or-more SecCertificateRef objects. The
# responsibility for freeing this CFArray will be with the caller. This
# CFArray must remain alive for the entire connection, so in practice it
# will be stored with a single SSLSocket, along with the reference to the
# keychain.
certificates = []
identities = []

# Filter out bad paths.
paths = (path for path in paths if path)

try:
    for file_path in paths:
        new_identities, new_certs = _load_items_from_file(keychain, file_path)
        identities.extend(new_identities)
        certificates.extend(new_certs)

    # Ok, we have everything. The question is: do we have an identity? If
    # not, we want to grab one from the first cert we have.
    if not identities:
        new_identity = Security.SecIdentityRef()
        status = Security.SecIdentityCreateWithCertificate(
            keychain, certificates[0], ctypes.byref(new_identity)
        )
        _assert_no_error(status)
        identities.append(new_identity)

        # We now want to release the original certificate, as we no longer
        # need it.
        CoreFoundation.CFRelease(certificates.pop(0))

    # We now need to build a new CFArray that holds the trust chain.
    trust_chain = CoreFoundation.CFArrayCreateMutable(
        CoreFoundation.kCFAllocatorDefault,
        0,
        ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
    )
    for item in itertools.chain(identities, certificates):
        # ArrayAppendValue does a CFRetain on the item. That's fine,
        # because the finally block will release our other refs to them.
        CoreFoundation.CFArrayAppendValue(trust_chain, item)

    return trust_chain
finally:
    for obj in itertools.chain(identities, certificates):
        CoreFoundation.CFRelease(obj)


