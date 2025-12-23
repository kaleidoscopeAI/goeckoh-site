def neuro_acoustic_mirror(input_text):
    corrected = correct_speech(input_text)
    print(f"[Echo in your voice]: {corrected}")
    return corrected

