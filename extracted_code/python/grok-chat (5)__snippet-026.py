self.vad_model,
threshold=0.45,                 # LOWER than default 0.5-0.6
                                # → catches quiet, monotone, low-energy autistic speech
                                # (many of us speak softly or flatly when masking or calm)

sampling_rate=16000,

min_silence_duration_ms=1200,   # 1.2 seconds (UP from 100-700ms)
                                # → gives us long thinking pauses without cutting us off
                                # (autistic processing delays, word-finding, scripting time)

speech_pad_ms=400,              # 0.4 seconds padding front/back (UP from 30-100ms)
                                # → includes our slow on-ramps and trailing “….” thoughts

min_speech_duration_ms=250,     # keep low – we sometimes say just one word at a time
