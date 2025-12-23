"""A thread that calls refresh() at regular intervals."""

def __init__(self, live: "Live", refresh_per_second: float) -> None:
    self.live = live
    self.refresh_per_second = refresh_per_second
    self.done = Event()
    super().__init__(daemon=True)

def stop(self) -> None:
    self.done.set()

def run(self) -> None:
    while not self.done.wait(1 / self.refresh_per_second):
        with self.live._lock:
            if not self.done.is_set():
                self.live.refresh()


