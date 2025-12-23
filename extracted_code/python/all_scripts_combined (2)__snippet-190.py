app = Flask(__name__)

@app.get("/api/metrics")
def metrics_api() -> Any:
    rows = _load_rows(config.paths.metrics_csv)
    return jsonify(rows)

@app.get("/api/voice-profile")
def voice_profile_api() -> Any:
    return jsonify(_voice_profile_counts(config))

@app.get("/api/strategies")
def strategies_api() -> Any:
    return jsonify(
        [
            {"category": s.category, "title": s.title, "description": s.description}
            for s in STRATEGIES
        ]
    )

@app.get("/api/behavior")
def behavior_api() -> Any:
    return jsonify(_recent_behavior_events(config))

@app.get("/api/guidance-events")
def guidance_events_api() -> Any:
    rows = _load_guidance(config.paths.guidance_csv)
    return jsonify(rows)

@app.get("/api/settings")
def settings_api() -> Any:
    if settings_store is None:
        return jsonify({})
    return jsonify(settings_store.get_settings())

@app.patch("/api/settings")
def settings_update() -> Any:
    if settings_store is None:
        return jsonify({"error": "settings store unavailable"}), 400
    payload = request.get_json(force=True, silent=True) or {}
    settings_store.update(
        correction_echo_enabled=payload.get("correction_echo_enabled"),
        support_voice_enabled=payload.get("support_voice_enabled"),
    )
    config.behavior.correction_echo_enabled = bool(settings_store.data.get("correction_echo_enabled", True))
    config.behavior.support_voice_enabled = bool(settings_store.data.get("support_voice_enabled", False))
    return jsonify(settings_store.get_settings())

@app.get("/api/support-phrases")
def support_phrases_api() -> Any:
    if settings_store is None:
        return jsonify([])
    return jsonify(settings_store.list_support_phrases())

@app.post("/api/support-phrases")
def add_support_phrase() -> Any:
    if settings_store is None:
        return jsonify({"error": "settings store unavailable"}), 400
    payload = request.get_json(force=True, silent=True) or {}
    phrase = payload.get("phrase")
    if not phrase:
        return jsonify({"error": "phrase missing"}), 400
    settings_store.add_support_phrase(phrase)
    return jsonify(settings_store.list_support_phrases())

@app.post("/record_facet")
def record_facet() -> Any:
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    if 'style' not in request.form:
        return jsonify({"error": "No style provided"}), 400

    audio_file = request.files['audio']
    style = request.form['style']

    if not audio_file.filename:
        return jsonify({"error": "No selected file"}), 400

    try:
        temp_dir = config.paths.voices_dir / "temp_facets"
        temp_dir.mkdir(exist_ok=True)
        temp_wav_path = temp_dir / f"{style}_{os.urandom(4).hex()}.wav"
        audio_file.save(temp_wav_path)

        data, sr = sf.read(temp_wav_path, dtype="float32")
        if sr != config.audio.sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=config.audio.sample_rate)

        voice_profile = VoiceProfile(audio=config.audio, base_dir=config.paths.voices_dir / "voice_profile")
        voice_profile.add_sample_from_wav(np.asarray(data, dtype=np.float32), style)

        os.remove(temp_wav_path)

    return jsonify({"status": "success", "message": f"Facet '{style}' recorded successfully."}), 200
    except Exception as e:
        print(f"Error recording facet: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/")
def index() -> str:
    rows = _load_rows(config.paths.metrics_csv)
    phrase_rows, timeline_labels, timeline_rates = _summarize_metrics(rows)
    guidance_rows = _load_guidance(config.paths.guidance_csv)

    recent_guidance = guidance_rows[-25:]

    total_attempts = sum(r["attempts"] for r in phrase_rows)
    total_corrections = sum(r["corrections"] for r in phrase_rows)
    overall_rate = (total_corrections / total_attempts) if total_attempts else 0.0

    featured_strategies = STRATEGIES[:8]

    return render_template_string(
        TEMPLATE,
        child_name=config.child_name,
        total_attempts=total_attempts,
        overall_rate=overall_rate,
        phrases=phrase_rows,
        phrase_labels=[r["phrase"] for r in phrase_rows],
        phrase_rates=[round(r["rate"] * 100.0, 1) for r in phrase_rows],
        timeline_labels=timeline_labels,
        timeline_rates=[round(v, 1) for v in timeline_rates],
        strategies=featured_strategies,
        guidance_events=recent_guidance,
    )

return app


