def __init__(self, enable_voice: bool = True):

    self.enable_voice = enable_voice and pyttsx3

    self.engine = None

    self.event_queue = Queue()

    self.running = False

    self.thread = None

    if self.enable_voice:

        try:

            self.engine = pyttsx3.init()

            self.engine.setProperty('rate', 150)

            logging.info("Text-to-speech initialized")

            self.running = True

            self.thread = threading.Thread(target=self._process_queue, daemon=True)

            self.thread.start()

        except Exception as e:

            logging.warning(f"Text-to-speech init failed: {e}. Voice disabled")

            self.enable_voice = False


def _process_queue(self):

    while self.running:

        try:

            event_type, confidence, details = self.event_queue.get(timeout=1.0)

            message = self._generate_message(event_type, confidence, details)

            if message and self.engine:

                self.engine.say(message)

                self.engine.runAndWait()

            self.event_queue.task_done()

        except QueueEmpty:

            continue

        except Exception as e:

            logging.error(f"Voice processing error: {e}")


def _generate_message(self, event_type: str, confidence: float, details: Dict[str, Any]) -> str:

    base = f"Confidence {confidence * 100:.0f} percent."

    messages = {

        "NODE_STRESS_HIGH": f"Node {details.get('node_id', 'unknown')} high stress: {details.get('stress', 0):.1f}. {base}",

        "NODE_ENERGY_LOW": f"Node {details.get('node_id', 'unknown')} low energy: {details.get('energy', 0):.1f}. {base}",

        "INSIGHT_DISCOVERED": f"New {details.get('type', 'general')} insight. {base}",

        "GLOBAL_STRESS_HIGH": f"System stress high: {details.get('avg_stress', 0):.1f}. {base}",

        "CRITICAL_SIM_ERROR": f"Error at step {details.get('step', 'unknown')}: {details.get('error', '')}. {base}",

        "AUTHENTICATION_SUCCESS": "Authentication successful.",

        "AUTHENTICATION_FAILED": "Authentication failed.",

        "SIM_START": "Simulation starting.",

        "SIM_END": "Simulation ended.",

        "NODE_ADMET_GOOD": f"Node {details.get('node_id', 'unknown')} good ADMET score: {details.get('admet_score', 0):.1f}. {base}"

    }

    return messages.get(event_type, "")


def log_event(self, event_type: str, confidence: float, details: Dict[str, Any]):

    if self.enable_voice:

        try:

            self.event_queue.put_nowait((event_type, confidence, details))

        except QueueFull:

            logging.warning("Voice queue full, event dropped")


def close(self):

    self.running = False

    if self.thread:

        self.thread.join(timeout=2)

        if self.thread.is_alive():

            logging.warning("Voice thread did not terminate")

    if self.engine:

        self.engine.stop()

        logging.info("Text-to-speech stopped")


