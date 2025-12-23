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

