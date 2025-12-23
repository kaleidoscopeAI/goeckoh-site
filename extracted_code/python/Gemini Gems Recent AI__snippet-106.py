def __init__(self, bank_id: Optional[str] = None, capacity: int = 100):

    self.bank_id = bank_id if bank_id else str(uuid.uuid4())

    self.capacity = capacity

    self.insights: List[Insight] = []

    self.lock = threading.Lock()


def add_insight(self, insight: Insight) -> bool:

    with self.lock:

        if len(self.insights) < self.capacity:

            self.insights.append(insight)

            logging.debug(f"Insight {insight.insight_id} added to MemoryBank {self.bank_id}")

            return True

        logging.warning(f"MemoryBank {self.bank_id} at capacity")

        return False


def retrieve_insights(self, num: int = 1) -> List[Insight]:

    with self.lock:

        if not self.insights:

            return []

        return random.sample(self.insights, min(num, len(self.insights)))


