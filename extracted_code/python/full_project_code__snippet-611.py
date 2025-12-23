def __init__(self, prog, indent_increment=2, max_help_position=16, width=None):
    if width is None:
        try:
            width = shutil.get_terminal_size().columns - 2
        except Exception:
            pass
    argparse.HelpFormatter.__init__(self, prog, indent_increment,
                                    max_help_position, width)


