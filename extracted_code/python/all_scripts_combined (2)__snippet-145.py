def create_app(companion: EchoCompanion) -> Flask:
    app = Flask(__name__)

    @app.route("/api/latest")
    def api_latest() -> Any:
        return jsonify(companion.get_latest_metrics())

    @app.route("/api/stats")
    def api_stats() -> Any:
        return jsonify(companion.get_phrase_stats())

    @app.route("/api/utterance", methods=["POST"])
    async def api_utterance() -> Any:
        try:
            wav_bytes = await request.get_data()
            if not wav_bytes:
                return jsonify({"error": "no data"}), 400
            audio_np = _decode_wav(wav_bytes)
            res = await companion.process_utterance_for_mobile(audio_np)
            return jsonify(res)
        except Exception as exc:
            print(f"⚠️ /api/utterance error: {exc}")
            return jsonify({"error": "internal error"}), 500

    @app.route("/")
    def index() -> Any:
        return render_template_string(DASHBOARD_HTML)

    return app


def _decode_wav(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    import wave

    wf = wave.open(buf, "rb")
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    wf.close()
    if sampwidth != 2:
        raise ValueError("expected 16-bit PCM")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    if framerate != 16_000:
        duration = len(audio) / framerate
        target_len = int(duration * 16_000)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio)
    return audio
