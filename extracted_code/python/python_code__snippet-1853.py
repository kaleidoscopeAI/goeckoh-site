old: str
new: str

def __rich__(self) -> Group:
    if WINDOWS:
        pip_cmd = f"{get_best_invocation_for_this_python()} -m pip"
    else:
        pip_cmd = get_best_invocation_for_this_pip()

    notice = "[bold][[reset][blue]notice[reset][bold]][reset]"
    return Group(
        Text(),
        Text.from_markup(
            f"{notice} A new release of pip is available: "
            f"[red]{self.old}[reset] -> [green]{self.new}[reset]"
        ),
        Text.from_markup(
            f"{notice} To update, run: "
            f"[green]{escape(pip_cmd)} install --upgrade pip"
        ),
    )


