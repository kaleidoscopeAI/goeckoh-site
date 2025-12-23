import asyncio
from concurrent.futures import ThreadPoolExecutor

from flask import Flask
from werkzeug.serving import run_simple

from companion import EchoCompanion
from dashboard import create_app


def main():
    companion = EchoCompanion()
    app = create_app(companion)

    # Use a ThreadPoolExecutor for blocking operations (like Flask's run_simple)
    # in an asyncio application.
    executor = ThreadPoolExecutor(max_workers=2)

    async def run_flask_and_loop():
        # Start the Flask app in a separate thread
        print("[Main] Starting Flask dashboard...")
        flask_task = asyncio.get_event_loop().run_in_executor(
            executor,
            run_simple,
            "0.0.0.0",  # Listen on all interfaces
            8765,
            app,
            use_reloader=False,  # Disable reloader for production
            threaded=True,
        )

        # Start the SpeechLoop
        print("[Main] Starting Echo Companion speech loop...")
        await companion.start_loop()

        # Keep the main task running until Flask or SpeechLoop stops
        await asyncio.gather(flask_task, companion.speech_loop._q.join()) # This is a placeholder, needs proper event handling

    try:
        asyncio.run(run_flask_and_loop())
    except KeyboardInterrupt:
        print("[Main] Shutting down...")
        asyncio.run(companion.stop_loop())
    finally:
        executor.shutdown(wait=True)


