is_container = True     # Backwards compatibility

@cached_property
def resources(self):
    return self.finder.get_resources(self)


