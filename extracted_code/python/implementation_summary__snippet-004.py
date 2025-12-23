from voice_logger import log_voice_characteristics

profile = log_voice_characteristics(
    audio_samples=[audio1, audio2, audio3],
    sr=22050,
    user_id="alice",
    output_dir="./bubble_data",
    speaker_embedding=None  # Optional
)
