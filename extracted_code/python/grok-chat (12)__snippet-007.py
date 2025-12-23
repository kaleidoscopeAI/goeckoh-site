def chatterbox_tts(text, ref_audio, emotion="neutral"):
    try:
        CHATTERBOX_MODEL.clone_voice(ref_audio)  # Zero-shot
        audio_out = CHATTERBOX_MODEL.synthesize(text, emotion=emotion)  # FP16 internal
        return audio_out.numpy()  # Assume returns torch tensor
    except Exception as e:
        logger.error(f"Chatterbox error: {e}. Fallback.")
        fallback_engine.say(text)
        fallback_engine.runAndWait()
        return np.array([])

