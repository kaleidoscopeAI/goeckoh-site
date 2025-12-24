Minimal Python AI-Agent scaffold for validating a system config and starting a simple REPL agent.

Quick start:
1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. python -m cli start

Commands:
- python -m cli validate  # validate config.yaml against config.schema.yaml
- python -m cli fix       # auto-fix common issues in config.yaml
- python -m cli start     # start a REPL with the agent

The default config.yaml includes guidance from the tools instructions.

Speak command (voice cloning required)
- This CLI enforces voice cloning. There is no fallback to local TTS.
- You MUST provide a clean WAV sample of your voice either via:
  - The config (voice_profile_path), or
  - The CLI (--voice-profile)
- The sample must be at least the minimum duration (default: 5 seconds) and sampled at 16kHz where possible.
- Example:
  - Record 5s and use recorded sample for cloning:
    python -m cli speak --record --duration 5 --voice-profile ./sample_voice.wav
  - Use an existing WAV sample and an input file to correct and play:
    python -m cli speak --input-file ./input.wav --voice-profile ./sample_voice.wav

Read documents
- Read all .txt/.md/.pdf files from a folder and its subfolders:
  python -m cli read-docs --path ./documents --recursive
- The command defaults to recursive reading. Use --no-recursive to restrict to the top-level folder only.
- Make sure the documents_path in config.yaml points to your folder if you don't pass --path.
- Subfolder contents are ingested and summarized automatically.

Read code
- Read all .py files from a folder and its subfolders:
  python -m cli read-code --path ./project --recursive
- The command defaults to recursive reading. Use --no-recursive to restrict to the top-level folder only.
- The command uses the same config.documents_path as a fallback if --path is not provided.

Notes:
- Voice cloning uses Coqui TTS and resemblyzer. The following packages are required — they are large and may require a GPU for fast synthesis:
  - TTS (Coqui TTS)
  - resemblyzer
- If the voice profile is missing, or the libraries cannot be initialized, the CLI will exit with an error.
- For accurate cloning, provide a clean voice sample (>= 5s) sampled at 16kHz WAV format.
