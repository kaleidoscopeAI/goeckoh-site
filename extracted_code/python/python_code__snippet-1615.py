"""Read bytes from a file while tracking progress.

Args:
    path (Union[str, PathLike[str], BinaryIO]): The path to the file to read, or a file-like object in binary mode.
    mode (str): The mode to use to open the file. Only supports "r", "rb" or "rt".
    buffering (int): The buffering strategy to use, see :func:`io.open`.
    encoding (str, optional): The encoding to use when reading in text mode, see :func:`io.open`.
    errors (str, optional): The error handling strategy for decoding errors, see :func:`io.open`.
    newline (str, optional): The strategy for handling newlines in text mode, see :func:`io.open`
    total: (int, optional): Total number of bytes to read. Must be provided if reading from a file handle. Default for a path is os.stat(file).st_size.
    description (str, optional): Description of task show next to progress bar. Defaults to "Reading".
    auto_refresh (bool, optional): Automatic refresh, disable to force a refresh after each iteration. Default is True.
    transient: (bool, optional): Clear the progress on exit. Defaults to False.
    console (Console, optional): Console to write to. Default creates internal Console instance.
    refresh_per_second (float): Number of times per second to refresh the progress information. Defaults to 10.
    style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
    complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
    finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
    pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
    disable (bool, optional): Disable display of progress.
    encoding (str, optional): The encoding to use when reading in text mode.

Returns:
    ContextManager[BinaryIO]: A context manager yielding a progress reader.

"""

columns: List["ProgressColumn"] = (
    [TextColumn("[progress.description]{task.description}")] if description else []
)
columns.extend(
    (
        BarColumn(
            style=style,
            complete_style=complete_style,
            finished_style=finished_style,
            pulse_style=pulse_style,
        ),
        DownloadColumn(),
        TimeRemainingColumn(),
    )
)
progress = Progress(
    *columns,
    auto_refresh=auto_refresh,
    console=console,
    transient=transient,
    get_time=get_time,
    refresh_per_second=refresh_per_second or 10,
    disable=disable,
)

reader = progress.open(
    file,
    mode=mode,
    buffering=buffering,
    encoding=encoding,
    errors=errors,
    newline=newline,
    total=total,
    description=description,
)
return _ReadContext(progress, reader)  # type: ignore[return-value, type-var]


