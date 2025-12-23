    import platform

    features = get_windows_console_features()
    from pip._vendor.rich import print

    print(f'platform="{platform.system()}"')
    print(repr(features))


