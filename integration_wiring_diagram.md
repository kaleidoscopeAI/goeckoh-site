# Echo-Crystal AGI Integration: Script-by-Script Wiring Diagram

## 1. Introduction

This document outlines a detailed, script-by-script plan for integrating the specialized therapeutic `echovoice` system with the general-purpose, self-evolving `cognitive_crystal_ai` (CCA) project. The goal is to create a hybrid AGI companion capable of advanced therapeutic interaction, emotional intelligence, and autonomous learning within a physical and digital environment.

## 2. Core Systems Overview

### 2.1. `echovoice` System (Unified `echo_prime.py`)

The `echo_prime.py` script encapsulates the `echovoice` functionality, including:
*   **Sensory Input:** Audio capture via `sounddevice`.
*   **Speech Processing:** `faster_whisper` (STT), `Coqui TTS` (Voice Crystal), `librosa` (Prosody).
*   **Emotional Core:** `CrystallineHeart` (arousal, valence).
*   **Therapeutic Logic:** `AbaEngine` (ABA strategies).
*   **LLM Integration:** `ollama` (for supportive responses/grammar correction).
*   **Organic Subconscious:** A simple `OrganicSubconscious` (node metabolism, replication).
*   **GUI Hooks:** `gui_callback` for `echo_gui.py`.

### 2.2. `cognitive_crystal_ai` (CCA) Project

The CCA project is a modular AGI framework with the following key components (inferred from `README.md` and directory structure):
*   **Backend (`cognitive_crystal_ai/backend/`):**
    *   `main.py`, `server.py`: Core application logic and web server (`Quart`).
    *   `core/`: Foundational logic, utilities.
    *   `data/`: Data models, storage interfaces.
    *   `engines/`: "Thought Engines" (e.g., Kaleidoscope, Perspective).
    *   `memory/`: "Crystalline Memory" implementation (`faiss-cpu` suggests vector memory).
    *   `utils/`: Helper functions.
*   **Organic AI Final (`cognitive_crystal_ai/organic_ai_final/`):**
    *   `main.py`, `organic_ai/`: Core "Organic AI" logic (node growth, DNA mutation).
    *   `adapters/`: Integration with external systems.
    *   `dash/`: Possible internal dashboard (`streamlit`).
    *   `pynput`, `pyautogui`: HID Emulation.
*   **Seed Crystal AGI (`cognitive_crystal_ai/seed_crystal_agi/`):**
    *   `seed_crystal.db`: Persistent state.
    *   `audio/`: Specialized audio processing (distinct from `echovoice`'s).
    *   `shapes/`: Visualization components.
    *   `state/`: State management for AGI seed.
*   **Frontend (`cognitive_crystal_ai/frontend/`):** React/Three.js web application for visualization and interaction.
*   **AGI Real (`cognitive_crystal_ai/agireal/`):** `onbrain_autonomous.py`, `onbrain_online_avatar.py` - specific AGI instantiations or modes.
*   **Scripts (`cognitive_crystal_ai/scripts/`):** `run_system.sh`, `setup.sh` - orchestration.

## 3. Integration Architecture - High Level

The `echo_prime.py` system, focused on immediate therapeutic interaction via speech, would become a specialized "avatar" or "interface layer" for the more generalized `cognitive_crystal_ai` AGI. The `cognitive_crystal_ai`'s backend would serve as the central cognitive processing unit, integrating data from `echo_prime.py` and other sensors, driving decisions, and orchestrating outputs.

```
+-------------------+                                +---------------------------+
|    EchoVoice      |          Bidirectional         |    Cognitive Crystal AI   |
| (echo_prime.py)   | <----------------------------> | (backend, organic_ai_final, |
|                   |          Data & Control          | frontend, seed_crystal_agi) |
+--------^----------+                                +-------------^-------------+
         |                                                         |
         | Sensory Input (Audio)                                   | HID Emulation (Physical World)
         | Speech Output                                           | Frontend Visualisation
         | ABA Interventions                                       | AGI Internal State
         v                                                         v
+-------------------+                                +---------------------------+
| Child / Therapist |                                |    External Environment   |
+-------------------+                                +---------------------------+
```

## 4. Detailed Script-by-Script Wiring

This section details the primary communication channels and functional responsibilities.

### 4.1. Sensory Input & Initial Processing

*   **`echo_prime.py` (EchoSystem.listen_loop -> EchoSystem.process_utterance)**
    *   `sounddevice` (Python library): Captures raw audio.
    *   `faster_whisper` (Python library): Transcribes audio to text.
    *   `CrystallineHeart` (in `echo_prime.py`): Performs initial audio-based emotional analysis (arousal/valence).
    *   **Data Out:**
        *   Text (User utterance)
        *   Emotional Intensity (arousal, valence)
        *   Raw Audio (for optional deeper analysis by CCA backend)

*   **`cognitive_crystal_ai/backend/server.py` (Quart API)**
    *   **Receives:** Text, Emotional Intensity, (optional) Raw Audio from `echo_prime.py`.
    *   **Mechanism:** `echo_prime.py` would send this data via HTTP POST requests to specific endpoints exposed by `backend/server.py`.
    *   **Routes:**
        *   `/api/sensory/audio`: For raw audio/features.
        *   `/api/sensory/text`: For transcribed text.
        *   `/api/sensory/emotion`: For arousal/valence updates.

### 4.2. Emotional & Organic Processing (Deep AGI Integration)

*   **`cognitive_crystal_ai/backend/server.py` (API -> Backend Logic)**
    *   **Data In:** Text, Emotional Intensity.
    *   **Flow:** Routes incoming data to relevant `backend` modules.
    *   `backend/core/emotion_processor.py` (hypothetical new script): Would receive `echo_prime.py`'s emotional data.
    *   `backend/memory/crystalline_memory.py`: Encodes text and emotional context into the AGI's memory.
    *   `backend/engines/thought_engines.py`: Receives text for analysis (e.g., semantic content, child's intent).

*   **`cognitive_crystal_ai/organic_ai_final/main.py` (Organic Core)**
    *   **Receives:** Emotional intensity/stimuli from `backend/server.py` (or directly from `echo_prime.py` if IPC is direct).
    *   **Mechanism:** Could be a dedicated WebSocket connection or API endpoint.
    *   **Function:** Manages `OrganicNode` metabolism, replication, and mutation based on processed emotional and cognitive stimuli.
    *   **Data Out:** Updates on Node Network state (number of nodes, energy levels, coherence).

### 4.3. Cognitive Processing & Decision Making

*   **`cognitive_crystal_ai/backend/engines/thought_engines.py`**
    *   **Data In:** Text, Emotional Context, Crystalline Memory state.
    *   **Function:**
        *   `Perspective Engine`: Analyzes child's perspective.
        *   `Kaleidoscope Engine`: Generates novel cognitive pathways.
    *   **LLM Integration:** `backend/engines/llm_reasoning.py` (hypothetical) uses Ollama for self-reflection, strategic planning, and context-aware response generation.
    *   **Data Out:** Therapeutic recommendations, response text, environmental adjustment commands.

*   **`cognitive_crystal_ai/agireal/onbrain_autonomous.py` (AGI Brain Core)**
    *   **Data In:** Outputs from `Thought Engines`, Organic Core state.
    *   **Function:** Central decision-making, guided by Master Hamiltonian. Prioritizes therapeutic goals while considering emergent Organic AI insights.
    *   **Data Out:** Final validated action plan (e.g., "speak this text with this prosody," "dim lights," "trigger ABA strategy").

### 4.4. Action Output & Feedback Loops

*   **Control Flow (AGI Brain -> EchoVoice):**
    *   `agireal/onbrain_autonomous.py` (or `backend/server.py` as a proxy) would send commands back to `echo_prime.py`.
    *   **Mechanism:** HTTP POST or WebSocket.
    *   **Commands:**
        *   `echo_prime.py` (EchoSystem.voice.speak): Text to speak, desired prosody/style.
        *   `echo_prime.py` (EchoSystem.aba.evaluate/trigger_strategy): Specific ABA intervention to execute.

*   **`echo_prime.py` (EchoSystem.voice.speak)**
    *   `VoiceCrystal` (in `echo_prime.py`): Synthesizes speech with prosody transfer using `Coqui TTS`.
    *   `sounddevice` (Python library): Plays audio to the child.
    *   **Data Out:** Child's response (audio -> sensory input), internal state changes.

*   **`cognitive_crystal_ai/organic_ai_final/main.py` (HID Emulation)**
    *   **Data In:** Environmental adjustment commands from `agireal/onbrain_autonomous.py` (or `backend/server.py`).
    *   **Mechanism:** Internal function calls or IPC if `organic_ai_final` runs as a separate process.
    *   **Function:** `pynput`, `pyautogui` control physical devices (e.g., adjust smart lighting, interact with computer applications).

*   **`cognitive_crystal_ai/frontend/src/` (React/Three.js Visualization)**
    *   **Data In:** Real-time internal state from `backend/server.py` (e.g., Node network, Crystalline Memory map, emotional radar, LLM monologue).
    *   **Mechanism:** WebSocket or SSE (Server-Sent Events) from `backend/server.py`.
    *   **Function:** Renders the "Cognitive Cube" and other visualizations for the caregiver.

### 4.5. Integration of GUIs

*   **Option 1: `echo_gui.py` as a `cognitive_crystal_ai/frontend` module.**
    *   The `echo_gui.py` (customtkinter) would be integrated as a sub-module or specialized view within the `cognitive_crystal_ai` React/Three.js frontend. This is complex and might require `customtkinter` to render to a web canvas, or a separate electron-like wrapper. This is less ideal.

*   **Option 2: `echo_gui.py` as a dedicated "Therapeutic Monitor" client.**
    *   `echo_gui.py` remains a separate `customtkinter` application.
    *   It communicates directly with `cognitive_crystal_ai/backend/server.py` via a dedicated API (e.g., WebSockets or REST endpoints for state updates and command injection).
    *   **Data In:** Real-time emotional intensity, ABA strategy logs, Organic AI node count/energy from `backend/server.py`.
    *   **Commands Out:** Manual intervention overrides from `echo_gui.py` to `backend/server.py`.
    *   **Flow:**
        `echo_gui.py` <---WebSocket/REST---> `backend/server.py` <---Internal API/IPC---> `agireal/onbrain_autonomous.py` / `echo_prime.py`.

    This option is generally more feasible, treating `echo_gui.py` as a specialized client.

## 5. Mind Map (Textual Representation of Wiring)

```
ECHO-CRYSTAL AGI (Integrated System)
├── ECHO_PRIME.PY (Therapeutic Interface & Sensing Layer)
│   ├── Inputs:
│   │   ├── Microphone (sounddevice)
│   │   └── Child's Speech
│   ├── Processing:
│   │   ├── VAD
│   │   ├── faster_whisper --> Transcribed Text
│   │   ├── CrystallineHeart (audio RMS) --> Emotional Intensity (local)
│   │   └── VoiceCrystal (Coqui TTS)
│   ├── Outputs:
│   │   ├── Speaker (voice.speak) --> Corrected/Therapeutic Speech
│   │   ├── HTTP POST / WebSocket to CCA Backend --> Text, Emotional Intensity, Raw Audio (optional)
│   │   └── GUI Callback to ECHO_GUI.PY
│   └── Internal Threads:
│       └── OrganicSubconscious (simplified, locally fed)
│
├── COGNITIVE_CRYSTAL_AI (Core Cognitive Processing & Control)
│   ├── BACKEND/ (Main Brain & API Gateway)
│   │   ├── SERVER.PY (Quart API)
│   │   │   ├── INPUT ROUTES:
│   │   │   │   ├── /api/sensory/text <-- from ECHO_PRIME.PY
│   │   │   │   ├── /api/sensory/emotion <-- from ECHO_PRIME.PY
│   │   │   │   ├── /api/sensory/audio <-- from ECHO_PRIME.PY (optional)
│   │   │   │   └── /api/environment/sensors <-- from External Sensors
│   │   │   ├── OUTPUT ROUTES:
│   │   │   │   ├── /api/commands/speech --> to ECHO_PRIME.PY (Text, Prosody/Style)
│   │   │   │   ├── /api/commands/aba --> to ECHO_PRIME.PY (ABA Strategy Trigger)
│   │   │   │   ├── /api/commands/hid --> to ORGANIC_AI_FINAL/MAIN.PY (HID Controls)
│   │   │   │   └── /api/state/updates --> to FRONTEND/ (for visualization)
│   │   │   ├── Backend GUI (cognitive_crystal_ai/backend/gui/): For debugging/admin
│   │   ├── MEMORY/CRYSTALLINE_MEMORY.PY
│   │   │   ├── Inputs: Transcribed Text, Emotional Context, Environmental Data, Outcomes
│   │   │   └── Outputs: Semantic Search Results, Context for Thought Engines
│   │   ├── ENGINES/THOUGHT_ENGINES.PY (Kaleidoscope, Perspective)
│   │   │   ├── Inputs: Crystalline Memory, Emotional State, Organic AI State
│   │   │   └── Outputs: Cognitive Insights, Therapeutic Recommendations
│   │   ├── ENGINES/LLM_REASONING.PY (Ollama)
│   │   │   ├── Inputs: Context from Thought Engines, Crystalline Memory
│   │   │   └── Outputs: Self-Reflection, Strategic Plans, Response Generation
│   │   └── DATA/, CORE/, UTILS/ (Supporting Modules)
│   │
│   ├── ORGANIC_AI_FINAL/ (Self-Evolving Learning & Physical Interface)
│   │   ├── ORGANIC_AI/MAIN.PY (Core Node Logic)
│   │   │   ├── Inputs: Emotional Stimuli, Cognitive Novelty from BACKEND/
│   │   │   └── Outputs: Node Network State, Emergent Strategies
│   │   └── ADAPTERS/, DASH/, HID_EMULATION.PY (Pynput/Pyautogui)
│   │       ├── Inputs: HID Commands from BACKEND/
│   │       └── Outputs: Physical Device Control, Environmental Adjustments
│   │
│   ├── SEED_CRYSTAL_AGI/ (Specialized AGI Instance/State Management)
│   │   ├── STATE/ (Persistent AGI State)
│   │   └── SEED_CRYSTAL.DB (Database)
│   │
│   ├── AGIREAL/ (Specialized AGI Modes/Avatars)
│   │   ├── ONBRAIN_AUTONOMOUS.PY (Main AGI Decision Maker)
│   │   │   ├── Inputs: Outputs from BACKEND Thought Engines, Organic AI, ABA
│   │   │   └── Outputs: Action Plan, Commands to ECHO_PRIME.PY, HID Emulation
│   │   └── ONBRAIN_ONLINE_AVATAR.PY (Specific AGI persona/interface)
│   │
│   └── FRONTEND/ (React/Three.js Web UI)
│       ├── Inputs: Real-time State from BACKEND/SERVER.PY (WebSockets/SSE)
│       └── Outputs: Cognitive Cube Visualization, Emotional Radar, Caregiver Input
│
└── ECHO_GUI.PY (Therapeutic Monitor - OPTIONAL CLIENT)
    ├── Inputs: Real-time State from BACKEND/SERVER.PY (WebSockets/REST)
    ├── Outputs: Manual Override Commands to BACKEND/SERVER.PY
    └── Renders: Live Transcript, Emotional Intensity Bar, ABA Log, Organic Node Count

```
