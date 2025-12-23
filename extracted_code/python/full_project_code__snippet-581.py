try:
    import pip._vendor.urllib3.contrib.pyopenssl as pyopenssl
    pyopenssl.inject_into_urllib3()
except ImportError:
    pass

