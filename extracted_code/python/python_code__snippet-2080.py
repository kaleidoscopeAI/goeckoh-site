def insert_option_group(
    self, idx: int, *args: Any, **kwargs: Any
) -> optparse.OptionGroup:
    """Insert an OptionGroup at a given position."""
    group = self.add_option_group(*args, **kwargs)

    self.option_groups.pop()
    self.option_groups.insert(idx, group)

    return group

@property
def option_list_all(self) -> List[optparse.Option]:
    """Get a list of all options, including those in option groups."""
    res = self.option_list[:]
    for i in self.option_groups:
        res.extend(i.option_list)

    return res


