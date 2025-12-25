import logging
import threading
import queue
import csv
import os
from datetime import datetime

# Setup a standard logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalLogger(threading.Thread):
    def __init__(self, csv_queue: queue.Queue):
        super().__init__(daemon=True)
        self.csv_queue = csv_queue
        
        log_dir = "clinical_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"session_{session_id}.csv")
        
        self._init_csv()

    def _init_csv(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "rms_energy", "gcl", "entropy", 
                "tts_latency_ms", "raw_input_text", "corrected_output_text"
            ])

    def run(self):
        while True:
            try:
                data_tuple = self.csv_queue.get()
                
                timestamp = datetime.now().isoformat()
                log_entry = [timestamp] + list(data_tuple)
                
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(log_entry)
                    
                self.csv_queue.task_done()
            except Exception as e:
                logger.error(f"ClinicalLogger failed to write to CSV: {e}")
