def register_finder(loader, finder_maker):
    _finder_registry[type(loader)] = finder_maker


