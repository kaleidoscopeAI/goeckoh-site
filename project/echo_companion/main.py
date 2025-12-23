import argparse
from pathlib import Path

from companion_loop import SpeechCompanion
from voice_clone_engine import VoiceCloneEngine
from config import SPEAKER_REF_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo Companion - offline voice clone loop")
    parser.add_argument(
        "--enroll",
        metavar="REF_DIR",
        help=f"Directory of speaker reference WAVs (default: {SPEAKER_REF_DIR})",
        default=None,
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run the real-time companion loop",
    )
    args = parser.parse_args()

    if args.enroll is not None:
        engine = VoiceCloneEngine()
        ref_dir = SPEAKER_REF_DIR if args.enroll == "default" else Path(args.enroll)
        engine.enroll_from_directory(ref_dir)
        print("Enrollment complete.")
        return

    if args.loop:
        companion = SpeechCompanion()
        companion.loop_forever()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
