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

    loop = SpeechLoop(config)
    if args.command == "record":
        cmd_record(loop, args)
    elif args.command == "list":
        cmd_list(config)
    elif args.command == "summary":
        cmd_summary(config)
    elif args.command == "run":
        cmd_run(loop)
    elif args.command == "dashboard":
        cmd_dashboard(config, args.host, args.port)
    elif args.command == "gui":
        cmd_gui(config)
    elif args.command == "strategies":
        cmd_strategies(args.category, args.event)
    elif args.command == "comfort":
        cmd_comfort(loop, args.event, args.message)
    elif args.command == "record-voice-facet":
        cmd_record_voice_facet(loop, args.style, args.seconds, args.name)
    elif args.command == "show-voice-profile":
        cmd_show_voice_profile(loop)


