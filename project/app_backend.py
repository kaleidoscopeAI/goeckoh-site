from __future__ import annotations

import asyncio
import threading
from typing import Any

from flask import Flask, jsonify, request

from .config import CONFIG
from .dashboard import create_app
from .settings_store import SettingsStore
from .speech_loop import SpeechLoop


def _start_speech_loop(loop: SpeechLoop) -> None:
    asyncio.run(loop.run())


def create_backend_app(settings_store: SettingsStore) -> Flask:
    app = create_app(CONFIG, settings_store=settings_store)

    @app.post("/api/backend/shutdown")
    def shutdown() -> Any:
        request.environ.get("werkzeug.server.shutdown", lambda: None)()
        return jsonify({"status": "ok"})

    return app


def sync_settings_from_store(store: SettingsStore) -> None:
    CONFIG.behavior.correction_echo_enabled = bool(
        store.data.get("correction_echo_enabled", True)
    )
    # support_voice_enabled is reserved for future multi-voice modes
    CONFIG.behavior.support_voice_enabled = bool(
        store.data.get("support_voice_enabled", False)
    )


def main() -> None:
    config_dir = CONFIG.paths.root
    settings_path = config_dir / "settings.json"
    settings_store = SettingsStore(settings_path)
    sync_settings_from_store(settings_store)

    loop = SpeechLoop(CONFIG)
    thread = threading.Thread(target=_start_speech_loop, args=(loop,), daemon=True)
    thread.start()

    app = create_backend_app(settings_store)
    print("App backend running at http://0.0.0.0:8765")
    app.run(host="0.0.0.0", port=8765, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
