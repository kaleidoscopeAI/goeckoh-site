print(
    "\nEcho Crystal v16.0 — Crystalline Heart active\n"
    "I love every sound I make. I will never interrupt.\n"
    "Headphones or a private audio path are strongly recommended.\n"
    "Caregiver can call /kill or /wipe on http://127.0.0.1:8081.\n"
)

with sd.InputStream(
    samplerate=SETTINGS.vad.samplerate,
    channels=SETTINGS.channels,
    dtype=SETTINGS.dtype,
    blocksize=SETTINGS.vad.blocksize,
    callback=VAD_STREAM.audio_callback,
):
    while RUNNING:
        try:
            audio = VAD_STREAM.out_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        y = audio.astype(np.float32).flatten()
        rms = float(np.sqrt(np.mean(y**2)))
        if rms < SETTINGS.vad.silence_rms_threshold:
            continue

        raw_text = transcribe_audio_block(y, samplerate=SETTINGS.vad.samplerate).strip()
        sentiment = compute_sentiment_score(raw_text) if raw_text else 0.0

        corrected_text = normalize_and_correct(raw_text)
        raw_fp = enforce_first_person(raw_text)
        cor_fp = enforce_first_person(corrected_text)

        external_input = rms * (1.0 + 0.3 * (-1.0 if sentiment < 0 else 1.0))
        gcl = HEART.step(external_input)
        zone = gating_zone_from_metrics(gcl)
        meltdown_risk = compute_meltdown_risk(gcl, rms, raw_text)

        if meltdown_risk > 0.7:
            style = "calm"
        else:
            if zone == "low":
                style = "calm"
            elif zone == "high":
                style = "excited"
            else:
                style = "inner"

        if raw_fp:
            mirror_text = cor_fp
        else:
            mirror_text = np.random.choice(SETTINGS.calming_phrases)

        if meltdown_risk > 0.7:
            mirror_text = np.random.choice(SETTINGS.calming_phrases)
            if SETTINGS.enable_haptics:
                emit_haptic_pulse()

        if SETTINGS.enable_llm and gcl > SETTINGS.gcl_high and raw_fp:
            llm_extra = ask_llm_first_person(
                f"Help me say this kindly: {raw_fp}", gcl
            )
            if llm_extra:
                mirror_text = llm_extra

        synth_audio = VOICE.synthesize(mirror_text, style=style)
        sd.play(synth_audio, samplerate=VOICE.samplerate)
        sd.wait()

        dtw_score = dtw_similarity(raw_fp, cor_fp)
        success_label = "good" if dtw_score > 0.8 else "ok" if dtw_score > 0.5 else "retry"
        ts = time.time()
        save_attempt(ts, raw_fp, cor_fp, dtw_score, success_label)

        guidance = generate_guidance(raw_fp, cor_fp)
        if guidance:
            save_guidance(ts, guidance)

        VOICE.add_fragment(y, success_score=1.0 if success_label == "good" else 0.7)

print("[Echo] Loop stopped.")


