"""A subprocess call failed."""

reference = "subprocess-exited-with-error"

def __init__(
    self,
    *,
    command_description: str,
    exit_code: int,
    output_lines: Optional[List[str]],
) -> None:
    if output_lines is None:
        output_prompt = Text("See above for output.")
    else:
        output_prompt = (
            Text.from_markup(f"[red][{len(output_lines)} lines of output][/]\n")
            + Text("".join(output_lines))
            + Text.from_markup(R"[red]\[end of output][/]")
        )

    super().__init__(
        message=(
            f"[green]{escape(command_description)}[/] did not run successfully.\n"
            f"exit code: {exit_code}"
        ),
        context=output_prompt,
        hint_stmt=None,
        note_stmt=(
            "This error originates from a subprocess, and is likely not a "
            "problem with pip."
        ),
    )

    self.command_description = command_description
    self.exit_code = exit_code

def __str__(self) -> str:
    return f"{self.command_description} exited with {self.exit_code}"


