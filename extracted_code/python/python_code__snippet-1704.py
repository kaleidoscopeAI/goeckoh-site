hChainEngine = None
hRootCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
try:
    # Add custom CA certs to an in-memory cert store
    for cert_bytes in custom_ca_certs:
        CertAddEncodedCertificateToStore(
            hRootCertStore,
            X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
            cert_bytes,
            len(cert_bytes),
            CERT_STORE_ADD_USE_EXISTING,
            None,
        )

    # Create a custom cert chain engine which exclusively trusts
    # certs from our hRootCertStore
    cert_chain_engine_config = CERT_CHAIN_ENGINE_CONFIG()
    cert_chain_engine_config.cbSize = sizeof(cert_chain_engine_config)
    cert_chain_engine_config.hExclusiveRoot = hRootCertStore
    pConfig = pointer(cert_chain_engine_config)
    phChainEngine = pointer(HCERTCHAINENGINE())
    CertCreateCertificateChainEngine(
        pConfig,
        phChainEngine,
    )
    hChainEngine = phChainEngine.contents

    # Get and verify a cert chain using the custom chain engine
    _get_and_verify_cert_chain(
        ssl_context,
        hChainEngine,
        hIntermediateCertStore,
        pPeerCertContext,
        pChainPara,
        server_hostname,
        chain_flags,
    )
finally:
    if hChainEngine:
        CertFreeCertificateChainEngine(hChainEngine)
    CertCloseStore(hRootCertStore, 0)


