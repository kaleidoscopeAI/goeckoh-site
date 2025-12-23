import logging
logging.basicConfig(filename='audit.log', level=logging.INFO)

def ethical_gate(action, confidence, reason):
    logging.info(f"TIME: {time.ctime()}, ACTION: {action}, CONFIDENCE: {confidence:.3f}, REASON: {reason}")
    return confidence > 0.85

# In device control
if ethical_gate("adjust_cpu", 0.92, "Normal operation"):
    apply_device_control()
else:
    print("Ethical gate blocked action")
