class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        if not torch or not GCNConv:
            logging.error("PyTorch or PyTorch Geometric missing, GNN disabled")
            self.is_enabled = False
            return
        self.is_enabled = True
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if not self.is_enabled:
            return None
        try:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        except Exception as e:
            logging.error(f"GNN forward failed: {e}")
            return None

class SelfCorrectionModel(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        if not torch:
            logging.error("PyTorch missing, VAE disabled")
            self.is_enabled = False
            return
        self.is_enabled = True
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        if not self.is_enabled:
            return None
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if not self.is_enabled:
            return None, None, None
        try:
            h = self.encoder(x)
            mu, log_var = h.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)
            return self.decoder(z), mu, log_var
        except Exception as e:
            logging.error(f"VAE forward failed: {e}")
            return None, None, None

class SystemVoice:
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

class VisualInputProcessor:
    def __init__(self, video_path: Optional[str] = None):
        self.video_path = video_path
        self.cap = None
        if cv2:
            try:
                self.cap = cv2.VideoCapture(video_path or 0)
                if not self.cap.isOpened():
                    raise IOError("Cannot open video source")
                logging.info(f"Visual processor initialized for {'webcam' if not video_path else video_path}")
            except Exception as e:
                logging.error(f"Visual processor init failed: {e}")
                self.cap = None
        else:
            logging.warning("OpenCV missing, visual input disabled")

    def get_frame(self) -> Optional[np.ndarray]:
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
            logging.warning("Failed to read frame")
        return None

    def process_frame(self) -> List[Insight]:
        frame = self.get_frame()
        if frame is None:
            return []
        if random.random() < 0.05:
            avg_color = np.mean(frame, axis=(0, 1)).tolist()
            return [Insight(type="visual_anomaly", data={"avg_color": avg_color}, confidence=0.7)]
        return []

    def close(self):
        if self.cap:
            self.cap.release()
            logging.info("Visual processor closed")

class WebSocketClient:
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

class QueryParser:
    def __init__(self):
        self.known_compounds = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        }

    def parse_query(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            query = query.lower()
            if "similar to" in query and "safer for the stomach" in query:
                for compound in self.known_compounds:
                    if compound in query:
                        return {
                            "type": "molecular_query",
                            "target_smiles": self.known_compounds[compound],
                            "constraints": {"logp": "< 3.0", "h_bond_donors": "< 3"}
                        }
            logging.warning(f"Unrecognized query: {query}")
            return None
        except Exception as e:
            logging.error(f"Query parsing failed: {e}")
            return None

