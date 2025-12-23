def audio_callback(indata, frames, time_info, status):
    global audio_buffer, is_speech, silence_start
    rms = np.sqrt(np.mean(indata**2))
    if rms > 0.01:  # voice detected
        audio_buffer.append(indata.copy())
        is_speech = True
        silence_start = None
    else:
        if is_speech and silence_start is None:
            silence_start = time.time()
        elif is_speech and silence_start and (time.time() - silence_start > 1.2):  # 1.2 s silence = end of utterance
            full_audio = np.concatenate(audio_buffer)
            audio_queue.put(full_audio)
            audio_buffer = []
            is_speech = False
            silence_start = None

