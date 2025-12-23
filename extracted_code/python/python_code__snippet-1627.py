"""Renders file size downloaded and total, e.g. '0.5/2.3 GB'.

Args:
    binary_units (bool, optional): Use binary units, KiB, MiB etc. Defaults to False.
"""

def __init__(
    self, binary_units: bool = False, table_column: Optional[Column] = None
) -> None:
    self.binary_units = binary_units
    super().__init__(table_column=table_column)

def render(self, task: "Task") -> Text:
    """Calculate common unit for completed and total."""
    completed = int(task.completed)

    unit_and_suffix_calculation_base = (
        int(task.total) if task.total is not None else completed
    )
    if self.binary_units:
        unit, suffix = filesize.pick_unit_and_suffix(
            unit_and_suffix_calculation_base,
            ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"],
            1024,
        )
    else:
        unit, suffix = filesize.pick_unit_and_suffix(
            unit_and_suffix_calculation_base,
            ["bytes", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"],
            1000,
        )
    precision = 0 if unit == 1 else 1

    completed_ratio = completed / unit
    completed_str = f"{completed_ratio:,.{precision}f}"

    if task.total is not None:
        total = int(task.total)
        total_ratio = total / unit
        total_str = f"{total_ratio:,.{precision}f}"
    else:
        total_str = "?"

    download_status = f"{completed_str}/{total_str} {suffix}"
    download_text = Text(download_status, style="progress.download")
    return download_text


