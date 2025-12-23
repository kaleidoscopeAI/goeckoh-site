from __future__ import annotations

import argparse
import asyncio

from .config import CompanionConfig, CONFIG
from .data_store import DataStore
from .speech_loop import SpeechLoop
from .reports import summarize
from .dashboard import create_app
from .calming_strategies import list_categories, by_category, suggest_for_event
from .guidance import GUIDANCE_SCRIPTS
from .advanced_voice_mimic import Style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autism speech companion control CLI")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="Record a canonical phrase")
    rec.add_argument("--text", required=True, help="Phrase text to associate with the recording")
    rec.add_argument("--seconds", type=float, default=3.0, help="Recording length")

    sub.add_parser("list", help="List known phrases")
    sub.add_parser("summary", help="Show metrics CSV path")

    run = sub.add_parser("run", help="Start the realtime loop")

    dash = sub.add_parser("dashboard", help="Launch therapist dashboard")
    dash.add_argument("--host", default="0.0.0.0")
    dash.add_argument("--port", type=int, default=8765)

    sub.add_parser("gui", help="Launch the local Tkinter caregiver/child dashboard")

    strat = sub.add_parser("strategies", help="Show calming strategies catalog")
    strat.add_argument("--category", choices=list_categories(), help="Filter by category")
    strat.add_argument(
        "--event",
        choices=[
            "meltdown",
            "transition",
            "anxious_speech",
            "anxious",
            "hyperactivity",
            "high_energy",
            "care_team_sync",
            "school_meeting",
            "communication_practice",
            "caregiver_reset",
            "interest_planning",
            "perseveration",
            "encouragement",
        ],
        help="Filter by event trigger",
    )

    comfort = sub.add_parser("comfort", help="Play a supportive prompt right now")
    comfort.add_argument("--event", choices=sorted(GUIDANCE_SCRIPTS.keys()), required=True)
    comfort.add_argument("--message", help="Override guidance text with a custom message")

    # New commands for Voice Crystal
    record_facet = sub.add_parser("record-voice-facet", help="Record a voice sample for a specific style (facet)")
    record_facet.add_argument("style", choices=["neutral", "calm", "excited"], help="Style label for this facet")
    record_facet.add_argument("--seconds", type=float, default=3.0, help="Duration of the recording in seconds")
    record_facet.add_argument("--name", help="Optional name suffix stored with the sample")

    sub.add_parser("show-voice-profile", help="Show the current VoiceProfile facets")

    return parser


def cmd_record(loop: SpeechLoop, args: argparse.Namespace) -> None:
    phrase = loop.record_phrase(args.text, args.seconds)
    print(f"Recorded phrase '{phrase.text}' as {phrase.audio_file}")


def cmd_list(config: CompanionConfig) -> None:
    data = DataStore(config)
    for phrase in data.list_phrases():
        print(f"{phrase.phrase_id}: '{phrase.text}' -> {phrase.audio_file}")


def cmd_summary(config: CompanionConfig) -> None:
    stats = summarize(config)
    if not stats:
        print("No attempts logged yet.")
        return
    print(f"Metrics CSV: {config.paths.metrics_csv}")
    for phrase_id, row in stats.items():
        rate = row.corrections / row.attempts if row.attempts else 0.0
        print(f"{phrase_id}: attempts={row.attempts} corrections={row.corrections} correction_rate={rate:.2f}")
    if config.paths.guidance_csv.exists():
        with config.paths.guidance_csv.open() as f:
            guidance_events = sum(1 for _ in f) - 1
        if guidance_events > 0:
            print(f"Guidance events logged: {guidance_events} (see {config.paths.guidance_csv})")


def cmd_run(loop: SpeechLoop) -> None:
    print("Starting realtime speech companion. Press Ctrl+C to exit.")
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_dashboard(config: CompanionConfig, host: str, port: int) -> None:
    app = create_app(config)
    print(f"Dashboard running at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


def cmd_gui(config: CompanionConfig) -> None:
    from .tk_gui import run_gui  # Lazy import to avoid tkinter dependency for non-GUI usage

    run_gui(config)


def cmd_strategies(category: str | None, event: str | None) -> None:
    if event:
        strategies = suggest_for_event(event)
    elif category:
        strategies = by_category(category)
    else:
        strategies = suggest_for_event("anxious_speech")
    print("Recommended calming strategies:")
    for strategy in strategies:
        cues = f" (cues: {', '.join(strategy.cues)})" if strategy.cues else ""
        print(f"- [{strategy.category}] {strategy.title}{cues}\n  {strategy.description}\n")


def cmd_comfort(loop: SpeechLoop, event: str, message: str | None) -> None:
    loop.coach.speak(event, override_text=message)


