#!/usr/bin/env python3
"""
Broken Speech Interpreter

This standalone script normalizes heavily abbreviated or "broken" speech,
extracts likely intent, and surfaces a short explanation.  Run with:

    python broken_speech_tool.py --text "u help me job money pls"

or launch interactive mode with no arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional audio / ML imports (only needed for --realtime)
try:
    import sounddevice as sd
    import soundfile as sf
    import whisper
    from TTS.api import TTS as CoquiTTS
except Exception:  # pragma: no cover - optional deps
    sd = None
    sf = None
    whisper = None
    CoquiTTS = None

# Common shorthand and shorthand-like word replacements.
CORRECTIONS: Dict[str, str] = {
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tx": "thanks",
    "tmrw": "tomorrow",
    "tmr": "tomorrow",
    "tho": "though",
    "luv": "love",
    "abt": "about",
    "bc": "because",
    "cuz": "because",
    "wanna": "want to",
    "gonna": "going to",
    "hafta": "have to",
    "im": "i am",
    "cant": "cannot",
    "dont": "do not",
    "idk": "i do not know",
    "smth": "something",
    "ppl": "people",
    "ya": "yeah",
    "omw": "on my way",
    "wtvr": "whatever",
    "r": "are",
    "n": "and",
    "w/": "with",
    "b4": "before",
    "lmk": "let me know",
    "asap": "as soon as possible",
    "msg": "message",
    "info": "information",
    "prob": "problem",
}

# Keyword groupings for super-lightweight intent detection.
INTENT_RULES: Sequence[Tuple[str, Sequence[str]]] = (
    ("request", ("please", "help", "need", "want", "can", "could", "assist", "support")),
    ("question", ("what", "why", "how", "when", "where", "who", "?", "do", "is", "are")),
    ("complaint", ("angry", "upset", "bad", "issue", "problem", "broke", "broken")),
    ("gratitude", ("thanks", "thank", "appreciate", "grateful")),
    ("greeting", ("hello", "hi", "hey", "morning", "afternoon")),
)

POSITIVE_WORDS = {
    "good",
    "great",
    "love",
    "happy",
    "awesome",
    "thanks",
    "appreciate",
    "success",
    "win",
}
NEGATIVE_WORDS = {
    "bad",
    "sad",
    "angry",
    "mad",
    "hate",
    "issue",
    "problem",
    "broke",
    "broken",
    "fail",
    "error",
}


@dataclass
class NormalizationResult:
    tokens: List[str]
    normalized_text: str
    notes: List[str]


def collapse_repeated_letters(word: str) -> str:
    """Collapse more than two repeated letters (heyyy -> heyy)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def normalize_text(text: str) -> NormalizationResult:
    notes: List[str] = []
    tokens: List[str] = []
    for raw_token in re.split(r"\s+", text.strip()):
        if not raw_token:
            continue
        cleaned = re.sub(r"[^\w'/]", "", raw_token.lower())
        if not cleaned:
            continue

        collapsed = collapse_repeated_letters(cleaned)
        if collapsed != cleaned:
            notes.append(f"Collapsed '{cleaned}' -> '{collapsed}'")

        replacement = CORRECTIONS.get(collapsed, collapsed)
        if replacement != collapsed:
            notes.append(f"Expanded '{collapsed}' -> '{replacement}'")

        tokens.extend(replacement.split())

    normalized_text = " ".join(tokens)
    return NormalizationResult(tokens=tokens, normalized_text=normalized_text, notes=notes)


def detect_intent(tokens: Sequence[str]) -> Tuple[str, float, List[str]]:
    best_intent = "unknown"
    best_score = 0.0
    evidence: List[str] = []
    token_set = list(tokens)

    for intent_name, keywords in INTENT_RULES:
        hits = [w for w in token_set if w in keywords or w.endswith("?")]
        if not hits:
            continue
        score = len(hits) / len(keywords)
        if score > best_score:
            best_score = score
            best_intent = intent_name
            evidence = hits

    return best_intent, round(best_score, 3), evidence


def detect_sentiment(tokens: Iterable[str]) -> str:
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def interpret(text: str) -> Dict[str, object]:
    normalization = normalize_text(text)
    intent, score, evidence = detect_intent(normalization.tokens)
    sentiment = detect_sentiment(normalization.tokens)

    interpretation = {
        "original_text": text.strip(),
        "normalized_text": normalization.normalized_text,
        "intent": intent,
        "intent_confidence": score,
        "intent_evidence": evidence,
        "sentiment": sentiment,
        "notes": normalization.notes,
    }
    return interpretation


def check_audio_deps() -> None:
    """Ensure optional audio/ML deps are present before entering realtime mode."""
    if any(x is None for x in (sd, sf, whisper, CoquiTTS)):
        print(
            "Real-time mode requires extra packages:\n"
            "  pip install openai-whisper sounddevice soundfile TTS\n"
        )
        sys.exit(1)


def run_realtime(args: argparse.Namespace) -> int:
    """
    Mic -> Whisper STT -> text normalization -> Coqui TTS (voice clone) -> playback.
    Keeps CLI behavior untouched when --realtime is not used.
    """
    check_audio_deps()

    if not args.voice_sample:
        print("Real-time mode requires --voice-sample /path/to/your_voice.wav")
        return 1
    if not os.path.exists(args.voice_sample):
        print(f"Voice sample not found: {args.voice_sample}")
        return 1

    print(f"Loading Whisper STT model: {args.whisper_model}")
    stt_model = whisper.load_model(args.whisper_model, device=args.device)

    print(f"Loading Coqui TTS model: {args.tts_model}")
    tts_model = CoquiTTS(
        model_name=args.tts_model,
        progress_bar=False,
        gpu=args.device.startswith("cuda"),
    )

    sample_rate = args.sample_rate
    print()
    print("Real-Time Broken Speech Voice Bridge")
    print("------------------------------------")
    print(f"- Mic sample rate: {sample_rate} Hz")
    print(f"- Chunk length:    {args.chunk_seconds} seconds")
    print(f"- Language:        {args.language}")
    print()
    print("Press ENTER, speak, then the system will normalize and speak back.")
    print("Ctrl+C to exit.")
    print()

    while True:
        try:
            input("Press ENTER to speak (Ctrl+C to exit)...")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting real-time mode.")
            break

        print("Recording...")
        audio = sd.rec(
            int(args.chunk_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Recording complete, transcribing...")

        in_path: Optional[str] = None
        out_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                in_path = f.name
                sf.write(in_path, audio, sample_rate)

            stt_result = stt_model.transcribe(in_path, language=args.language)
            original_text = (stt_result.get("text") or "").strip()
            if not original_text:
                print("[No speech detected]")
                continue

            interp = interpret(original_text)
            normalized = interp["normalized_text"].strip() or original_text

            print(f"Heard:      {original_text}")
            print(f"Corrected:  {normalized}")
            print(f"Intent:     {interp['intent']} ({interp['intent_confidence']})")
            print(f"Sentiment:  {interp['sentiment']}")
            print("Speaking back...")

            out_path = in_path.replace(".wav", "_out.wav")
            tts_model.tts_to_file(
                text=normalized,
                speaker_wav=args.voice_sample,
                language=args.language,
                file_path=out_path,
            )
            data, out_sr = sf.read(out_path, dtype="float32")
            sd.play(data, out_sr)
            sd.wait()
            print()
        finally:
            for path in (in_path, out_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def run_cli(args: argparse.Namespace) -> int:
    if args.text:
        result = interpret(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print("Broken Speech Interpreter (type 'exit' to quit)")
    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = interpret(user_input)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpret text or run a real-time speech bridge.")
    parser.add_argument(
        "-t",
        "--text",
        help="Text to interpret. If omitted, interactive mode is started.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable real-time microphone mode with TTS playback in your own voice.",
    )
    parser.add_argument(
        "--voice-sample",
        help="Path to a short audio file of YOUR voice (for cloning in real-time mode).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper STT model size: tiny, base, small, medium, large (default: base).",
    )
    parser.add_argument(
        "--tts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model name (default: tts_models/multilingual/multi-dataset/xtts_v2).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate (default: 16000).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=4.0,
        help="Recording chunk length in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Spoken language code for STT/TTS (default: en).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for STT/TTS: 'cpu' or 'cuda' (default: cpu).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.realtime:
        return run_realtime(args)
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
