        import _frozen_importlib_external as _fi
    except ImportError:
        import _frozen_importlib as _fi
    _finder_registry[_fi.SourceFileLoader] = ResourceFinder
    _finder_registry[_fi.FileFinder] = ResourceFinder
    # See issue #146
    _finder_registry[_fi.SourcelessFileLoader] = ResourceFinder
    del _fi
