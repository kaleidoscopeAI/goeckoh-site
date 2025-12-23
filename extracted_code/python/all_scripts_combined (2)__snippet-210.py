"""
Load settings from disk if available, otherwise fall back to defaults.
"""

if config_path is None:
    config_path = DEFAULT_ROOT / "config.json"

settings = SystemSettings()
if config_path.exists():
    data = json.loads(config_path.read_text(encoding="utf-8"))
    _apply_json(settings, data)

settings.paths.ensure_logs()
return settings


