def __init__(self, uri: str = "ws://localhost:8765", enable: bool = False):

    self.uri = uri

    self.enable = enable and websocket

    self.ws = None

    self.thread = None

    self.queue = Queue()

    if self.enable:

        logging.info(f"WebSocket initialized for {uri}")

    else:

        logging.info("WebSocket disabled")


def _run(self):

    def on_message(ws, message):

        try:

            self.queue.put_nowait(message)

            logging.debug(f"WS message received: {message[:50]}...")

        except QueueFull:

            logging.warning("WS queue full")

    def on_error(ws, error):

        logging.error(f"WS error: {error}")

    def on_close(ws, code, msg):

        logging.info("WS closed")

    def on_open(ws):

        logging.info("WS opened")

    try:

        self.ws = websocket.WebSocketApp(self.uri, on_open=on_open, on_message=on_message,

                                         on_error=on_error, on_close=on_close)

        self.ws.run_forever()

    except Exception as e:

        logging.error(f"WS connection failed: {e}")

        self.enable = False


def connect(self):

    if self.enable and not self.thread:

        self.thread = threading.Thread(target=self._run, daemon=True)

        self.thread.start()

        logging.info("WS thread started")


def receive(self) -> Optional[str]:

    if self.enable:

        try:

            return self.queue.get_nowait()

        except QueueEmpty:

            return None

    return None


def close(self):

    if self.enable and self.ws:

        self.ws.close()

        logging.info("WS closed")


