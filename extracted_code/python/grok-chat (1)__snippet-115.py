def auditory_motor_core(input_text, metrics):
    corrected = input_text.replace("you", "I").replace("your", "my").capitalize()
    print(f"[Cloned Echo]: {corrected}")
    stimulus = len(input_text) / 10.0
    success = len(corrected) > len(input_text) / 2  # Sim success
    metrics.register_attempt(success)
    return corrected, stimulus

