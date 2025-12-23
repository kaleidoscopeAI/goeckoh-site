from bubble_synthesizer import feed_text_through_bubble

audio, controls = feed_text_through_bubble(
    text="Hello world",
    profile=profile,
    dt=0.01
)
