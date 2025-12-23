Dependency & Function Map
=========================

> Quick reference of how the key modules in this repository fit together.

Runtime orchestration
---------------------
- `speech_loop.py` – Central real‑time loop. Wires together audio I/O (`audio_io.AudioIO`), STT/grammar (`speech_processing.SpeechProcessor`), persistence (`data_store.DataStore`), similarity scoring (`similarity.SimilarityScorer`), behaviour tracking (`behavior_monitor.BehaviorMonitor`), ABA advice (`calming_strategies.StrategyAdvisor`), voice synthesis (`advanced_voice_mimic.VoiceCrystal` + `voice_mimic.VoiceMimic`), inner guidance (`inner_voice.InnerVoiceEngine`), safety policy (`policy.GuardianPolicy`), and the cognitive stack (Heart + Brain + AGI). Optional bridge: `cca_bridge.CCABridgeClient` for remote cognitive backend commands.
- `cli.py` – User-facing entrypoint that selects between realtime, simulation, or seed modes and spins up `SpeechLoop` or related services. (Legacy shims: `speechinterventionsystem.py`, `math.py`, `jackson_companion_v15_3.py`.)
- `ws_server.py` / `server.py` – WebSocket/HTTP surfaces for dashboards; typically wrap the same seed/loop objects.

Core cognition
--------------
- `heart_core.py:EchoCrystallineHeart` – Torch emotional lattice + optional local LLM “sentience port”. Provides `step(full_audio, transcript)` returning arousal/valence/coherence plus optional LLM text. Utility: `enforce_first_person` for pronoun safety.
- `heart.py:CrystallineHeart` – Numpy ODE variant (used by legacy `speech_loop_legacy.py`). Also includes a Torch `EchoCrystallineHeart` aligned with `HeartSettings`.
- `core.py:CrystalBrain` – Hash-based embedding store + energetics/phi proxies. Uses `store.MemoryStore` for persistence and `math` helpers for coherence metrics.
- `agent.py:KQBCAgent` – Thin wrapper around `unified_system_agi_core.AGISystem`; evaluates utterance similarity and steps the AGI, exposing an `AGIStatus` snapshot.
- `trigger_engine.py`, `phenotyping_tracker.py`, `behavior_monitor.py`, `calming_strategies.py` – Safety/ABA sidecars fed by `SpeechLoop` to detect perseveration, meltdowns, and recommend calming text.

Speech & audio path
-------------------
- `audio_io.py` – Microphone / playback helpers plus RMS utilities.
- `speech_processing.py` – Handles STT (Whisper/faster-whisper) and simple grammar normalization; returns `(raw, corrected)` pairs.
- `text_utils.py` – Normalization + string similarity helpers shared across scoring and policy checks.
- `prosody.py` – Extracts F0/RMS envelopes from waveforms (`extract_prosody`) and can transfer pitch/energy onto synthesized TTS audio (`apply_prosody_to_tts`).
- `advanced_voice_mimic.py` + `voice_mimic.py` – Neural TTS/voice cloning frontends. `VoiceCrystal` coordinates TTS engine selection, profile loading, and playback surface used by `SpeechLoop` and `InnerVoiceEngine`.
- `inner_voice.py` – Runs “inner echo” guidance using `VoiceCrystal`, optionally logging attempts via `DataStore`.

Data & storage
--------------
- `data_store.py` – Filesystem-backed store for phrases, attempts, and phenotyping metrics; used heavily by `SpeechLoop`.
- `store.py` / `memory_store` in `core.py` – Persist embeddings and energetics snapshots for the CrystalBrain.
- `paths.py`, `config.py`, `settings.py` – Configuration dataclasses that define directory layout, audio/heart/LLM settings, and runtime flags consumed by almost every other module.

Lower-level utilities
---------------------
- `gears.py` – Lightweight message bus (`Message`, `GearNode`, `GearFabric`) plus immutable payload types (`Information`, `AudioData`, `SpeechData`, `EmotionData`, `AgentDecision`) for passing state between gears in the organic seed.
- `guidance.py`, `policy.py`, `ethics.py` – Safety/policy helpers invoked during coaching or when loading guardian policies.
- `events.py` – Shared event/metric dataclasses (`EchoEvent`, `HeartMetrics`, `BrainMetrics`, `now_ts`) passed between the Heart, Brain, and loop.
- `similarity.py` – Audio comparison helpers (voice match scoring).
- `cca_bridge.py` (CCABridgeClient) – Sends sensory packets outbound and polls commands for integration with a remote Cognitive Crystal AI backend.

Execution flow (high level)
---------------------------
1. CLI selects mode and instantiates `SpeechLoop`.
2. `SpeechLoop` pulls config/state, starts `AudioIO`, processors, voice stack, and optional backend bridges.
3. Incoming audio chunks → `SpeechProcessor` (STT/grammar) + `SimilarityScorer` + `BehaviorMonitor`.
4. Event feeds into `EchoCrystallineHeart` and `CrystalBrain`; AGI updates via `KQBCAgent`.
5. Voice output synthesized with `VoiceCrystal`/`VoiceMimic`, optionally guided by `InnerVoiceEngine` and policy checks.
6. Attempts/phrases persisted via `DataStore`; snapshots exposed to UI/WS servers.

Notable helper functions/classes
--------------------------------
- `heart_core.enforce_first_person(text)` – Ensures child-facing utterances stay in first person.
- `audio_io.AudioIO.rms` – Core silence/vad threshold support.
- `speech_loop.SpeechLoop.record_phrase` / `handle_chunk` – Public API for recording and scoring practice attempts.
- `agent.KQBCAgent.evaluate_correction` – Determines when to prompt a correction based on text similarity.
- `prosody.apply_prosody_to_tts` – Optional prosody transfer when cloning voices.
