import logging
logging.basicConfig(filename='audit.log', level=logging.INFO)

def ethical_gate(action, confidence, reason):
    logging.info(f"TIME: {time.ctime()}, ACTION: {action}, CONFIDENCE: {confidence:.3f}, REASON: {reason}")
    return confidence > 0.85

class UnifiedOrganicAI:
    # ...
    async def run_organic_cycle(self, sensor_input=None, web_input=None):
        # ...
        action = "optimize_hardware"
        confidence = 0.9  # Obtain from LLM or anomaly score in real code
        reason = "Routine optimization"
        if ethical_gate(action, confidence, reason):
            apply_device_controls()
        else:
            logging.warning("Ethical gate blocked action")
        # ...
