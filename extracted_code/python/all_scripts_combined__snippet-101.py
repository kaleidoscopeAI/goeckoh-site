def _load_rows(csv_path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_guidance(csv_path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def _voice_profile_counts(config: CompanionConfig) -> Dict[str, int]:
    base = config.paths.voices_dir / "voice_profile"
    counts: Dict[str, int] = {}
    for style in ("neutral", "calm", "excited"):
        style_dir = base / style
        counts[style] = len(list(style_dir.glob("*.wav"))) if style_dir.exists() else 0
    return counts


def _recent_behavior_events(config: CompanionConfig, limit: int = 50) -> List[Dict[str, Any]]:
    rows = _load_guidance(config.paths.guidance_csv)
    if not rows:
        return []
    return rows[-limit:]


def _summarize_metrics(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[float]]:
    """Return (phrase_rows, timeline_labels, timeline_rates)."""
    phrase_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"phrase": "Unknown", "attempts": 0, "corrections": 0})
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "corrections": 0})

    for row in rows:
        pid = row.get("phrase_id") or "Unknown"
        phrase_text = row.get("phrase_text") or pid
        needs_correction = (row.get("needs_correction") == "1")

        pstats = phrase_stats[pid]
        pstats["phrase"] = phrase_text
        pstats["attempts"] += 1
        if needs_correction:
            pstats["corrections"] += 1

        ts = row.get("timestamp_iso") or ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                date_key = dt.date().isoformat()
            except Exception:
                date_key = ts.split("T", 1)[0]
            dstats = daily[date_key]
            dstats["attempts"] += 1
            if needs_correction:
                dstats["corrections"] += 1

    phrase_rows: List[Dict[str, Any]] = []
    total_attempts = 0
    total_corrections = 0

    for stats in phrase_stats.values():
        attempts = stats["attempts"]
        corrections = stats["corrections"]
        rate = (corrections / attempts) if attempts else 0.0
        phrase_rows.append(
            {
                "phrase": stats["phrase"],
                "attempts": attempts,
                "corrections": corrections,
                "rate": rate,
            }
        )
        total_attempts += attempts
        total_corrections += corrections

    phrase_rows.sort(key=lambda r: r["attempts"], reverse=True)

    timeline_labels: List[str] = []
    timeline_rates: List[float] = []
    for date_key in sorted(daily.keys()):
        attempts = daily[date_key]["attempts"]
        corrections = daily[date_key]["corrections"]
        rate = (corrections / attempts) if attempts else 0.0
        timeline_labels.append(date_key)
        timeline_rates.append(rate * 100.0)

    return phrase_rows, timeline_labels, timeline_rates


def create_app(config: CompanionConfig = CONFIG, settings_store: SettingsStore | None = None) -> Flask:
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


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=8765, debug=False)


