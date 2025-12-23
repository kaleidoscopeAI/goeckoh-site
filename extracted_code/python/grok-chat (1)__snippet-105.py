def simulate_audio_input():
    return input("Speak: ").strip()

def correct_to_first_person(text):
    text = text.replace("you", "I").replace("your", "my").capitalize()
    return text

