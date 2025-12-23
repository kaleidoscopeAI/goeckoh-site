"""Command-line entry point for the Echo speech companion + AGI stack."""

from __future__ import annotations

import argparse
import asyncio

from .config import CompanionConfig, CONFIG
from .data_store import DataStore
from .speech_loop import SpeechLoop
from .reports import summarize
from .calming_strategies import (
    list_categories,
    by_category,
    suggest_for_event,
)
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

    sub.add_parser("seed", help="Start Organic Seed + websocket server")

    run = sub.add_parser("run", help="Start the realtime speech companion loop")
    sub.add_parser("mirror", help="Echo back speech in the child's own voice (mirror-only)")

    sim = sub.add_parser(
        "simulate",
        help="Run a synthetic loop (no microphone) to exercise logs + AGI",
    )

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


def cmd_simulate(config: CompanionConfig) -> None:
    print("Starting simulated speech loop (no microphone). Press Ctrl+C to exit.")
    # Import lazily to avoid requiring audio deps unless simulate is used.
    from .speech_loop_alt import SimulatedSpeechLoop

    sim_loop = SimulatedSpeechLoop(config)
    try:
        asyncio.run(sim_loop.run())
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_dashboard(config: CompanionConfig, host: str, port: int) -> None:
    from .dashboard import create_app

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


# New commands for Voice Crystal
def cmd_record_voice_facet(loop: SpeechLoop, style: Style, seconds: float, name: str | None) -> None:
    print(f"Recording {seconds:.1f}s for '{style}' facet...")
    audio = loop.audio_io.record_phrase(seconds)
    path = loop.voice_profile.add_sample_from_wav(audio, style, name=name)
    print(f"Saved facet at {path}")


def cmd_show_voice_profile(loop: SpeechLoop) -> None:
    print("Current Voice Profile Facets:")
    total = 0
    for style in ("neutral", "calm", "excited"):
        samples = loop.voice_profile.samples.get(style, [])
        print(f"- {style}: {len(samples)} sample(s)")
        for sample in samples:
            print(f"    • {sample.path.name} (score={sample.quality_score:.2f})")
        total += len(samples)
    if total == 0:
        print("  No facets recorded yet.")


def main(config: CompanionConfig = CONFIG) -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "record":
        loop = SpeechLoop(config)
        cmd_record(loop, args)
    elif args.command == "list":
        cmd_list(config)
    elif args.command == "summary":
        cmd_summary(config)
    elif args.command == "seed":
        from . import main as seed_app

        print("Starting Organic Seed websocket server. Press Ctrl+C to exit.")
        try:
            asyncio.run(seed_app.main())
        except KeyboardInterrupt:
            print("Stopped.")
    elif args.command == "run":
        loop = SpeechLoop(config)
        cmd_run(loop)
    elif args.command == "mirror":
        loop = SpeechLoop(config, mirror_only=True)
        cmd_run(loop)
    elif args.command == "simulate":
        cmd_simulate(config)
    elif args.command == "dashboard":
        cmd_dashboard(config, args.host, args.port)
    elif args.command == "gui":
        cmd_gui(config)
    elif args.command == "strategies":
        cmd_strategies(args.category, args.event)
    elif args.command == "comfort":
        loop = SpeechLoop(config)
        cmd_comfort(loop, args.event, args.message)
    elif args.command == "record-voice-facet":
        loop = SpeechLoop(config)
        cmd_record_voice_facet(loop, args.style, args.seconds, args.name)
    elif args.command == "show-voice-profile":
        loop = SpeechLoop(config)
        cmd_show_voice_profile(loop)


if __name__ == "__main__":
    main()
