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


