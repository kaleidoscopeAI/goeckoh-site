import click
import yaml
from pathlib import Path
from agent import Agent, validate_config, ValidationError, load_yaml

# NEW: imports for the voice pipeline (assumes all modules exist)
from speech_io import record_to_wav, playback_wav, get_wav_info
from stt import transcribe_file
from correction import correct_text
from tts import synthesize_text_to_wav

# NEW: import typing for CLI help
from typing import Optional

CONFIG_PATH = Path(__file__).parent / "config.yaml"

@click.group()
def cli():
    pass

@cli.command()
def validate():
    """Validate the YAML config against the schema."""
    config = load_yaml(CONFIG_PATH)
    try:
        validate_config(config)
        click.echo("Config is valid.")
    except ValidationError as e:
        click.echo(f"Config validation failed:\n{e}")

@cli.command()
def fix():
    """Attempt to auto-fix common issues in the config."""
    config = load_yaml(CONFIG_PATH)
    changed = False
    if "system_prompt" not in config or not config["system_prompt"].strip():
        config["system_prompt"] = "Updated default system prompt."
        changed = True
    if "tools" not in config or not isinstance(config["tools"], list):
        config["tools"] = []
        changed = True
    if "tracing" not in config:
        config["tracing"] = True
        changed = True
    if "log_level" not in config:
        config["log_level"] = "INFO"
        changed = True
    if changed:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        click.echo("Config auto-fixed and saved.")
    else:
        click.echo("No changes required. Config was already valid (or fixed).")

@cli.command()
def start():
    """Start a simple REPL agent using the provided config."""
    config = load_yaml(CONFIG_PATH)
    try:
        agent = Agent(config)
    except ValidationError as e:
        click.echo(f"Failed to start agent. Config validation failed:\n{e}")
        return
    click.echo("Agent started. Type messages (Ctrl-C to quit).")
    try:
        while True:
            msg = click.prompt("You")
            response = agent.respond(msg)
            click.echo(response)
    except (EOFError, KeyboardInterrupt):
        click.echo("Shutting down agent.")

@cli.command("read-docs")
@click.option("--path", "path_", type=click.Path(exists=True, file_okay=False), help="Path to documents folder to ingest.")
@click.option("--recursive/--no-recursive", default=True, help="Whether to recursively read subfolders (default: true)")
def read_docs(path_, recursive):
    """Read all .txt, .md and .pdf documents from a folder and ingest them into the agent."""
    config = load_yaml(CONFIG_PATH)
    documents_path = path_ or config.get("documents_path")
    if not documents_path:
        click.echo("Error: No documents path supplied via --path or config 'documents_path'.")
        return
    try:
        agent = Agent(config)
    except ValidationError as e:
        click.echo(f"Failed to start agent. Config validation failed:\n{e}")
        return
    try:
        summaries = agent.ingest_documents(documents_path, recursive=recursive)
        if not summaries:
            click.echo("No documents found in: " + str(documents_path))
            return
        click.echo(f"Ingested {len(summaries)} documents:")
        for s in summaries:
            click.echo(f"- {s['path']} ({s['length']} chars): {s['snippet']}")
    except Exception as e:
        click.echo(f"Failed to ingest documents: {e}")

@cli.command("read-code")
@click.option("--path", "path_", type=click.Path(exists=True, file_okay=False), help="Path to code folder to ingest.")
@click.option("--recursive/--no-recursive", default=True, help="Whether to recursively read subfolders (default: true)")
def read_code(path_, recursive):
    """Read all .py files from a folder and ingest them into the agent."""
    config = load_yaml(CONFIG_PATH)
    code_path = path_ or config.get("documents_path")  # reuse documents_path if code-specific path is not set
    if not code_path:
        click.echo("Error: No code path supplied via --path or config 'documents_path'.")
        return
    try:
        agent = Agent(config)
    except ValidationError as e:
        click.echo(f"Failed to start agent. Config validation failed:\n{e}")
        return
    try:
        summaries = agent.ingest_documents(code_path, recursive=recursive, extensions=[".py"])
        if not summaries:
            click.echo("No .py files found in: " + str(code_path))
            return
        click.echo(f"Ingested {len(summaries)} python files:")
        for s in summaries:
            click.echo(f"- {s['path']} ({s['length']} chars): {s['snippet']}")
    except Exception as e:
        click.echo(f"Failed to ingest code files: {e}")

@cli.command()
@click.option("--record", is_flag=True, help="Record audio from microphone")
@click.option("--duration", default=5.0, help="Recording duration in seconds (if --record)")
@click.option("--input-file", type=click.Path(exists=True), help="Path to an input WAV file")
@click.option("--voice-profile", type=click.Path(exists=True), help="Path to a WAV file with a sample of the user's voice for cloning")
@click.option("--output-file", type=click.Path(), default=None, help="Where to save the synthesized corrected audio (WAV)")
@click.option("--no-playback", is_flag=True, help="Don't play back output audio")
def speak(record, duration, input_file, voice_profile, output_file, no_playback):
    """Record or transcribe an audio file, correct it, synthesize in user's voice (cloning required) and play back."""
    config = load_yaml(CONFIG_PATH)
    enable_clone = config.get("enable_voice_clone", False)
    voice_profile_in_config = config.get("voice_profile_path")
    min_profile_duration = config.get("min_voice_profile_duration", 5.0)

    # Voice cloning is required
    if not enable_clone:
        click.echo("Error: Voice cloning is not enabled in the config. Set 'enable_voice_clone: true' to continue.")
        return

    profile = voice_profile or voice_profile_in_config
    if not profile:
        click.echo("Error: Voice cloning is enabled; you must provide a voice profile via --voice-profile or 'voice_profile_path' in config.")
        return

    if not Path(profile).exists():
        click.echo(f"Error: Provided voice profile path does not exist: {profile}")
        return

    # Validate the voice profile duration (requires soundfile)
    try:
        info = get_wav_info(profile)
    except Exception as e:
        click.echo(f"Error reading voice profile WAV: {e}")
        return

    if info.get("duration", 0) < float(min_profile_duration):
        click.echo(f"Error: Voice profile must be at least {min_profile_duration} seconds long. Provided sample is {info.get('duration', 0):.2f}s.")
        return

    try:
        if record:
            click.echo(f"Recording {duration}s from default microphone...")
            audio_path = record_to_wav(duration=duration)
        elif input_file:
            audio_path = input_file
        else:
            click.echo("Either --record or --input-file must be provided.")
            return

        click.echo("Transcribing...")
        text = transcribe_file(audio_path)
        click.echo(f"Transcribed: {text}")

        click.echo("Applying grammar corrections...")
        corrected = correct_text(text)
        click.echo(f"Corrected text: {corrected}")

        click.echo("Synthesizing corrected text (voice cloning)...")
        out_wav = synthesize_text_to_wav(corrected, out_wav=output_file, voice_profile_wav=profile, use_voice_clone=True)
        click.echo(f"Synthesized audio saved to: {out_wav}")

        if not no_playback:
            click.echo("Playing back (in your voice)...")
            playback_wav(out_wav)

    except ValidationError as e:
        click.echo(f"Config validation failed:\n{e}")
    except Exception as e:
        click.echo(f"Error: {e}")
