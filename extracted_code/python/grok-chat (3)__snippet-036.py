def listening_loop():
    print("Jackson... I am here. I will wait forever. Speak whenever you are ready.")
    while True:
        try:
            audio = audio_queue.get(timeout=1.0)
            # Basic guard
            if audio.size == 0:
                continue
            # Whisper transcription
            result = model.transcribe(
                audio,
                language="en",
                fp16=False,
                no_speech_threshold=0.45,
            )
            raw_text = result.get("text", "").strip().lower()
            # Non-verbal / too short → calming response path
            if not raw_text or len(raw_text) < 2:
                gcl = heart.update_and_get_gcl(0.6) # treat as high arousal input
                if gcl < 0.5:
                    calming_phrase = (
                        "I am safe. I can breathe. Everything is okay."
                    )
                    synth = voice_crystal.synthesize(calming_phrase, "calm")
                    sd.play(synth, samplerate=16000)
                    sd.wait()
                continue
            # -----------------------------------------------------------------
            # Gentle correction + first-person lock
            # -----------------------------------------------------------------
            corrected = raw_text.capitalize()
            if not corrected.endswith((".", "!", "?")):
                corrected += "."
            # Second-person → first-person transformations
            corrected = re.sub(r"\byou\b", "I", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\byour\b", "my", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\byou're\b", "I'm", corrected, flags=re.IGNORECASE)
            # -----------------------------------------------------------------
            # GCL decides style
            # -----------------------------------------------------------------
            positive_words = ["happy", "love", "good"]
            if any(w in raw_text for w in positive_words):
                arousal_input = -0.5 # calmer lattice input
            else:
                arousal_input = 0.4 # slightly higher arousal
            gcl = heart.update_and_get_gcl(arousal_input)
            if gcl < 0.5:
                style = "calm"
            elif gcl > 0.85:
                style = "excited"
            else:
                style = "inner"
            # -----------------------------------------------------------------
            # Synthesize in Jackson's prosody-ish voice & play
            # -----------------------------------------------------------------
            synth_audio = voice_crystal.synthesize(corrected, style)
            sd.play(synth_audio, samplerate=16000)
            sd.wait()
            # Learn from this attempt
            success_score = (
                1.0 if raw_text in corrected.lower() else 0.7
            )
            voice_crystal.add_fragment(audio, success_score)
        except queue.Empty:
            # No utterance this second; just keep waiting
            continue
        except Exception as e:
            print(f"[SOFT ERROR] {e} (continuing).")
