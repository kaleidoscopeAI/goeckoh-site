def transcribe_file(wav_path: str) -> str:
    """
    Transcribe a WAV file to text.
    Uses SpeechRecognition (Google Web Speech) by default. If local models are available,
    you can implement those as an alternative.
    """
    try:
        import speech_recognition as sr
    except Exception as e:
        raise RuntimeError("Install speechrecognition: pip install SpeechRecognition") from e

    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError("Speech recognition request failed: " + str(e))
