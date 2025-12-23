from goeckoh.psychoacoustic_engine.bubble_synthesizer import feed_text_through_bubble

audio, controls = feed_text_through_bubble(
    text="Hello world",
    profile=profile,
    vocoder_backend=my_vocoder,  # Real TTS backend
    dt=0.01  # 10ms frames
)
