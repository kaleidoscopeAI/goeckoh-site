"""Renders an auto-updating progress bar(s).

Args:
    console (Console, optional): Optional Console instance. Default will an internal Console instance writing to stdout.
    auto_refresh (bool, optional): Enable auto refresh. If disabled, you will need to call `refresh()`.
    refresh_per_second (Optional[float], optional): Number of times per second to refresh the progress information or None to use default (10). Defaults to None.
    speed_estimate_period: (float, optional): Period (in seconds) used to calculate the speed estimate. Defaults to 30.
    transient: (bool, optional): Clear the progress on exit. Defaults to False.
    redirect_stdout: (bool, optional): Enable redirection of stdout, so ``print`` may be used. Defaults to True.
    redirect_stderr: (bool, optional): Enable redirection of stderr. Defaults to True.
    get_time: (Callable, optional): A callable that gets the current time, or None to use Console.get_time. Defaults to None.
    disable (bool, optional): Disable progress display. Defaults to False
    expand (bool, optional): Expand tasks table to fit width. Defaults to False.
"""

def __init__(
    self,
    *columns: Union[str, ProgressColumn],
    console: Optional[Console] = None,
    auto_refresh: bool = True,
    refresh_per_second: float = 10,
    speed_estimate_period: float = 30.0,
    transient: bool = False,
    redirect_stdout: bool = True,
    redirect_stderr: bool = True,
    get_time: Optional[GetTimeCallable] = None,
    disable: bool = False,
    expand: bool = False,
) -> None:
    assert refresh_per_second > 0, "refresh_per_second must be > 0"
    self._lock = RLock()
    self.columns = columns or self.get_default_columns()
    self.speed_estimate_period = speed_estimate_period

    self.disable = disable
    self.expand = expand
    self._tasks: Dict[TaskID, Task] = {}
    self._task_index: TaskID = TaskID(0)
    self.live = Live(
        console=console or get_console(),
        auto_refresh=auto_refresh,
        refresh_per_second=refresh_per_second,
        transient=transient,
        redirect_stdout=redirect_stdout,
        redirect_stderr=redirect_stderr,
        get_renderable=self.get_renderable,
    )
    self.get_time = get_time or self.console.get_time
    self.print = self.console.print
    self.log = self.console.log

@classmethod
def get_default_columns(cls) -> Tuple[ProgressColumn, ...]:
    """Get the default columns used for a new Progress instance:
       - a text column for the description (TextColumn)
       - the bar itself (BarColumn)
       - a text column showing completion percentage (TextColumn)
       - an estimated-time-remaining column (TimeRemainingColumn)
    If the Progress instance is created without passing a columns argument,
    the default columns defined here will be used.

    You can also create a Progress instance using custom columns before
    and/or after the defaults, as in this example:

        progress = Progress(
            SpinnerColumn(),
            *Progress.default_columns(),
            "Elapsed:",
            TimeElapsedColumn(),
        )

    This code shows the creation of a Progress display, containing
    a spinner to the left, the default columns, and a labeled elapsed
    time column.
    """
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

@property
def console(self) -> Console:
    return self.live.console

@property
def tasks(self) -> List[Task]:
    """Get a list of Task instances."""
    with self._lock:
        return list(self._tasks.values())

@property
def task_ids(self) -> List[TaskID]:
    """A list of task IDs."""
    with self._lock:
        return list(self._tasks.keys())

@property
def finished(self) -> bool:
    """Check if all tasks have been completed."""
    with self._lock:
        if not self._tasks:
            return True
        return all(task.finished for task in self._tasks.values())

def start(self) -> None:
    """Start the progress display."""
    if not self.disable:
        self.live.start(refresh=True)

def stop(self) -> None:
    """Stop the progress display."""
    self.live.stop()
    if not self.console.is_interactive:
        self.console.print()

def __enter__(self) -> "Progress":
    self.start()
    return self

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    self.stop()

def track(
    self,
    sequence: Union[Iterable[ProgressType], Sequence[ProgressType]],
    total: Optional[float] = None,
    task_id: Optional[TaskID] = None,
    description: str = "Working...",
    update_period: float = 0.1,
) -> Iterable[ProgressType]:
    """Track progress by iterating over a sequence.

    Args:
        sequence (Sequence[ProgressType]): A sequence of values you want to iterate over and track progress.
        total: (float, optional): Total number of steps. Default is len(sequence).
        task_id: (TaskID): Task to track. Default is new task.
        description: (str, optional): Description of task, if new task is created.
        update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.

    Returns:
        Iterable[ProgressType]: An iterable of values taken from the provided sequence.
    """
    if total is None:
        total = float(length_hint(sequence)) or None

    if task_id is None:
        task_id = self.add_task(description, total=total)
    else:
        self.update(task_id, total=total)

    if self.live.auto_refresh:
        with _TrackThread(self, task_id, update_period) as track_thread:
            for value in sequence:
                yield value
                track_thread.completed += 1
    else:
        advance = self.advance
        refresh = self.refresh
        for value in sequence:
            yield value
            advance(task_id, 1)
            refresh()

def wrap_file(
    self,
    file: BinaryIO,
    total: Optional[int] = None,
    *,
    task_id: Optional[TaskID] = None,
    description: str = "Reading...",
) -> BinaryIO:
    """Track progress file reading from a binary file.

    Args:
        file (BinaryIO): A file-like object opened in binary mode.
        total (int, optional): Total number of bytes to read. This must be provided unless a task with a total is also given.
        task_id (TaskID): Task to track. Default is new task.
        description (str, optional): Description of task, if new task is created.

    Returns:
        BinaryIO: A readable file-like object in binary mode.

    Raises:
        ValueError: When no total value can be extracted from the arguments or the task.
    """
    # attempt to recover the total from the task
    total_bytes: Optional[float] = None
    if total is not None:
        total_bytes = total
    elif task_id is not None:
        with self._lock:
            total_bytes = self._tasks[task_id].total
    if total_bytes is None:
        raise ValueError(
            f"unable to get the total number of bytes, please specify 'total'"
        )

    # update total of task or create new task
    if task_id is None:
        task_id = self.add_task(description, total=total_bytes)
    else:
        self.update(task_id, total=total_bytes)

    return _Reader(file, self, task_id, close_handle=False)

@typing.overload
def open(
    self,
    file: Union[str, "PathLike[str]", bytes],
    mode: Literal["rb"],
    buffering: int = -1,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    *,
    total: Optional[int] = None,
    task_id: Optional[TaskID] = None,
    description: str = "Reading...",
) -> BinaryIO:
    pass

@typing.overload
def open(
    self,
    file: Union[str, "PathLike[str]", bytes],
    mode: Union[Literal["r"], Literal["rt"]],
    buffering: int = -1,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    *,
    total: Optional[int] = None,
    task_id: Optional[TaskID] = None,
    description: str = "Reading...",
) -> TextIO:
    pass

def open(
    self,
    file: Union[str, "PathLike[str]", bytes],
    mode: Union[Literal["rb"], Literal["rt"], Literal["r"]] = "r",
    buffering: int = -1,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    *,
    total: Optional[int] = None,
    task_id: Optional[TaskID] = None,
    description: str = "Reading...",
) -> Union[BinaryIO, TextIO]:
    """Track progress while reading from a binary file.

    Args:
        path (Union[str, PathLike[str]]): The path to the file to read.
        mode (str): The mode to use to open the file. Only supports "r", "rb" or "rt".
        buffering (int): The buffering strategy to use, see :func:`io.open`.
        encoding (str, optional): The encoding to use when reading in text mode, see :func:`io.open`.
        errors (str, optional): The error handling strategy for decoding errors, see :func:`io.open`.
        newline (str, optional): The strategy for handling newlines in text mode, see :func:`io.open`.
        total (int, optional): Total number of bytes to read. If none given, os.stat(path).st_size is used.
        task_id (TaskID): Task to track. Default is new task.
        description (str, optional): Description of task, if new task is created.

    Returns:
        BinaryIO: A readable file-like object in binary mode.

    Raises:
        ValueError: When an invalid mode is given.
    """
    # normalize the mode (always rb, rt)
    _mode = "".join(sorted(mode, reverse=False))
    if _mode not in ("br", "rt", "r"):
        raise ValueError("invalid mode {!r}".format(mode))

    # patch buffering to provide the same behaviour as the builtin `open`
    line_buffering = buffering == 1
    if _mode == "br" and buffering == 1:
        warnings.warn(
            "line buffering (buffering=1) isn't supported in binary mode, the default buffer size will be used",
            RuntimeWarning,
        )
        buffering = -1
    elif _mode in ("rt", "r"):
        if buffering == 0:
            raise ValueError("can't have unbuffered text I/O")
        elif buffering == 1:
            buffering = -1

    # attempt to get the total with `os.stat`
    if total is None:
        total = stat(file).st_size

    # update total of task or create new task
    if task_id is None:
        task_id = self.add_task(description, total=total)
    else:
        self.update(task_id, total=total)

    # open the file in binary mode,
    handle = io.open(file, "rb", buffering=buffering)
    reader = _Reader(handle, self, task_id, close_handle=True)

    # wrap the reader in a `TextIOWrapper` if text mode
    if mode in ("r", "rt"):
        return io.TextIOWrapper(
            reader,
            encoding=encoding,
            errors=errors,
            newline=newline,
            line_buffering=line_buffering,
        )

    return reader

def start_task(self, task_id: TaskID) -> None:
    """Start a task.

    Starts a task (used when calculating elapsed time). You may need to call this manually,
    if you called ``add_task`` with ``start=False``.

    Args:
        task_id (TaskID): ID of task.
    """
    with self._lock:
        task = self._tasks[task_id]
        if task.start_time is None:
            task.start_time = self.get_time()

def stop_task(self, task_id: TaskID) -> None:
    """Stop a task.

    This will freeze the elapsed time on the task.

    Args:
        task_id (TaskID): ID of task.
    """
    with self._lock:
        task = self._tasks[task_id]
        current_time = self.get_time()
        if task.start_time is None:
            task.start_time = current_time
        task.stop_time = current_time

def update(
    self,
    task_id: TaskID,
    *,
    total: Optional[float] = None,
    completed: Optional[float] = None,
    advance: Optional[float] = None,
    description: Optional[str] = None,
    visible: Optional[bool] = None,
    refresh: bool = False,
    **fields: Any,
) -> None:
    """Update information associated with a task.

    Args:
        task_id (TaskID): Task id (returned by add_task).
        total (float, optional): Updates task.total if not None.
        completed (float, optional): Updates task.completed if not None.
        advance (float, optional): Add a value to task.completed if not None.
        description (str, optional): Change task description if not None.
        visible (bool, optional): Set visible flag if not None.
        refresh (bool): Force a refresh of progress information. Default is False.
        **fields (Any): Additional data fields required for rendering.
    """
    with self._lock:
        task = self._tasks[task_id]
        completed_start = task.completed

        if total is not None and total != task.total:
            task.total = total
            task._reset()
        if advance is not None:
            task.completed += advance
        if completed is not None:
            task.completed = completed
        if description is not None:
            task.description = description
        if visible is not None:
            task.visible = visible
        task.fields.update(fields)
        update_completed = task.completed - completed_start

        current_time = self.get_time()
        old_sample_time = current_time - self.speed_estimate_period
        _progress = task._progress

        popleft = _progress.popleft
        while _progress and _progress[0].timestamp < old_sample_time:
            popleft()
        if update_completed > 0:
            _progress.append(ProgressSample(current_time, update_completed))
        if (
            task.total is not None
            and task.completed >= task.total
            and task.finished_time is None
        ):
            task.finished_time = task.elapsed

    if refresh:
        self.refresh()

def reset(
    self,
    task_id: TaskID,
    *,
    start: bool = True,
    total: Optional[float] = None,
    completed: int = 0,
    visible: Optional[bool] = None,
    description: Optional[str] = None,
    **fields: Any,
) -> None:
    """Reset a task so completed is 0 and the clock is reset.

    Args:
        task_id (TaskID): ID of task.
        start (bool, optional): Start the task after reset. Defaults to True.
        total (float, optional): New total steps in task, or None to use current total. Defaults to None.
        completed (int, optional): Number of steps completed. Defaults to 0.
        visible (bool, optional): Enable display of the task. Defaults to True.
        description (str, optional): Change task description if not None. Defaults to None.
        **fields (str): Additional data fields required for rendering.
    """
    current_time = self.get_time()
    with self._lock:
        task = self._tasks[task_id]
        task._reset()
        task.start_time = current_time if start else None
        if total is not None:
            task.total = total
        task.completed = completed
        if visible is not None:
            task.visible = visible
        if fields:
            task.fields = fields
        if description is not None:
            task.description = description
        task.finished_time = None
    self.refresh()

def advance(self, task_id: TaskID, advance: float = 1) -> None:
    """Advance task by a number of steps.

    Args:
        task_id (TaskID): ID of task.
        advance (float): Number of steps to advance. Default is 1.
    """
    current_time = self.get_time()
    with self._lock:
        task = self._tasks[task_id]
        completed_start = task.completed
        task.completed += advance
        update_completed = task.completed - completed_start
        old_sample_time = current_time - self.speed_estimate_period
        _progress = task._progress

        popleft = _progress.popleft
        while _progress and _progress[0].timestamp < old_sample_time:
            popleft()
        while len(_progress) > 1000:
            popleft()
        _progress.append(ProgressSample(current_time, update_completed))
        if (
            task.total is not None
            and task.completed >= task.total
            and task.finished_time is None
        ):
            task.finished_time = task.elapsed
            task.finished_speed = task.speed

def refresh(self) -> None:
    """Refresh (render) the progress information."""
    if not self.disable and self.live.is_started:
        self.live.refresh()

def get_renderable(self) -> RenderableType:
    """Get a renderable for the progress display."""
    renderable = Group(*self.get_renderables())
    return renderable

def get_renderables(self) -> Iterable[RenderableType]:
    """Get a number of renderables for the progress display."""
    table = self.make_tasks_table(self.tasks)
    yield table

def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
    """Get a table to render the Progress display.

    Args:
        tasks (Iterable[Task]): An iterable of Task instances, one per row of the table.

    Returns:
        Table: A table instance.
    """
    table_columns = (
        (
            Column(no_wrap=True)
            if isinstance(_column, str)
            else _column.get_table_column().copy()
        )
        for _column in self.columns
    )
    table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)

    for task in tasks:
        if task.visible:
            table.add_row(
                *(
                    (
                        column.format(task=task)
                        if isinstance(column, str)
                        else column(task)
                    )
                    for column in self.columns
                )
            )
    return table

def __rich__(self) -> RenderableType:
    """Makes the Progress class itself renderable."""
    with self._lock:
        return self.get_renderable()

def add_task(
    self,
    description: str,
    start: bool = True,
    total: Optional[float] = 100.0,
    completed: int = 0,
    visible: bool = True,
    **fields: Any,
) -> TaskID:
    """Add a new 'task' to the Progress display.

    Args:
        description (str): A description of the task.
        start (bool, optional): Start the task immediately (to calculate elapsed time). If set to False,
            you will need to call `start` manually. Defaults to True.
        total (float, optional): Number of total steps in the progress if known.
            Set to None to render a pulsing animation. Defaults to 100.
        completed (int, optional): Number of steps completed so far. Defaults to 0.
        visible (bool, optional): Enable display of the task. Defaults to True.
        **fields (str): Additional data fields required for rendering.

    Returns:
        TaskID: An ID you can use when calling `update`.
    """
    with self._lock:
        task = Task(
            self._task_index,
            description,
            total,
            completed,
            visible=visible,
            fields=fields,
            _get_time=self.get_time,
            _lock=self._lock,
        )
        self._tasks[self._task_index] = task
        if start:
            self.start_task(self._task_index)
        new_task_index = self._task_index
        self._task_index = TaskID(int(self._task_index) + 1)
    self.refresh()
    return new_task_index

def remove_task(self, task_id: TaskID) -> None:
    """Delete a task if it exists.

    Args:
        task_id (TaskID): A task ID.

    """
    with self._lock:
        del self._tasks[task_id]


