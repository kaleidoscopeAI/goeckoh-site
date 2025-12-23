async def process_buffer():
    global buffer
    if len(buffer) < SAMPLE_RATE * 0.3: return  # 0.3s min
    start = time.time()
    try:
        denoised = wiener(buffer)
        signal_power = np.mean(denoised**2)
        noise_power = np.mean((buffer - denoised)**2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        logger.info(f"SNR: {snr_db:.2f} dB")

        segments, _ = WHISPER_MODEL.transcribe(denoised.astype(np.float32) / np.max(np.abs(denoised)))
        text = " ".join([seg.text for seg in segments]).strip()

        if not text: return

        corrected = correct_speech(text)
        logger.info(f"Original: {text} -> Corrected: {corrected}")

        gcl = heart.update_and_get_gcl(0.0)
        emotion = "calm" if gcl < 0.6 else "neutral"

        audio_out = chatterbox_tts(corrected, REF_SPEAKER_PATH, emotion)

        if len(audio_out) > 0:
            sd.play(audio_out, SAMPLE_RATE)
            sd.wait()

        lat = (time.time() - start) * 1000
        logger.info(f"Latency: {lat:.2f}ms")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        buffer = np.array([])

def audio_callback(indata, frames, time_info, status):
    global buffer
    chunk = indata.flatten()
    rms = np.mean(np.abs(chunk))
    heart.update_and_get_gcl(rms)
    if rms > NOISE_FLOOR + VAD_THRESHOLD:
        buffer = np.append(buffer, chunk)

def input_loop():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCKSIZE, callback=audio_callback):
        while running:
            if len(buffer) > SAMPLE_RATE * 0.3:
                asyncio.run_coroutine_threadsafe(process_buffer(), loop)
            time.sleep(0.001)  # Tightest

