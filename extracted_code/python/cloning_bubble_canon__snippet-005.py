from goeckoh.psychoacoustic_engine.voice_logger import log_voice_characteristics

profile = log_voice_characteristics(
    audio_samples=[y1, y2, y3],  # List of numpy arrays
    sr=22050,
    user_id="alice",
    output_dir="./bubble_data",
    speaker_embedding=encoder.encode(audio)  # From OpenVoice/SpeechBrain
)
