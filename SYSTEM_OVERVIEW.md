# System Overview - What This System Does

## 🎯 Primary Purpose

**Goeckoh Neuro-Acoustic Exocortex** is a **therapeutic voice therapy and AGI system** designed for **Auditory Prediction Error (APE) therapy**, particularly optimized for **autistic individuals**. It provides real-time voice feedback, speech correction, and therapeutic interventions through an advanced AI system.

## 🧠 Core Functionality

### 1. **Real-Time Voice Processing Pipeline**
The system implements a **6-step voice-to-clone pipeline**:

1. **Audio Capture** → Records user's voice from microphone
2. **Speech-to-Text (STT)** → Transcribes speech using Vosk or Sherpa-ONNX
3. **Linguistic Correction** → Corrects speech patterns and grammar
4. **Voice Synthesis (TTS)** → Generates corrected speech using neural TTS
5. **Prosody Transfer** → Transfers user's voice characteristics (pitch, rhythm, energy)
6. **Audio Playback** → Plays back the corrected speech in the user's own voice

### 2. **Voice Cloning & Mirroring**
- **Auto-captures** the user's voice on first utterance
- **Clones** the user's voice using neural TTS (Coqui TTS)
- **Mirrors** speech back with corrections while maintaining the user's voice characteristics
- **Adapts** voice over time through the "Voice Crystal" system

### 3. **Autism-Optimized Features**

#### Voice Activity Detection (VAD)
- **1.2 second pause tolerance** - Allows for processing pauses common in autistic speech
- **Respects natural speech patterns** - Doesn't interrupt during pauses
- **Lower threshold** (0.45) for quiet/monotone speech patterns

#### ABA Therapeutics Integration
- **Evidence-based interventions** for autism support
- **Positive reinforcement** system
- **Social stories** for emotional regulation
- **Skill tracking** across categories:
  - Self-care (brush teeth, wash hands, etc.)
  - Communication (greet others, ask for help, etc.)
  - Social Theory of Mind (share toys, understand emotions, etc.)

### 4. **Emotional Regulation System**

#### Crystalline Heart (1024-Node Lattice)
- **Emotional state modeling** using a 1024-node neural lattice
- **Global Coherence Level (GCL)** - Measures emotional stability
- **Real-time emotional regulation** through quantum-coupled dynamics
- **8D emotional state** tracking (joy, fear, trust, anger, anticipation, anxiety, focus, overwhelm)

#### Therapeutic Feedback
- **Co-regulation** - System adjusts its response based on user's emotional state
- **Crisis detection** - Identifies high-stress states and provides calming interventions
- **Sensory regulation** - Monitors and responds to sensory overwhelm

### 5. **Advanced AI Architecture**

#### Echo V4 Core
- **PsiState** - Unified state representation (world, body, self-model, history, emotion, quantum)
- **Hybrid Gated Dual-Core Architecture**:
  - **Fast Path**: Real-time therapeutic mirroring (< 100ms latency)
  - **Slow Path**: Deeper cognitive reasoning for complex tasks

#### Quantum Systems
- **Hamiltonian dynamics** for quantum state evolution
- **Molecular quantum systems** for advanced computation
- **Pure NumPy implementation** (no external dependencies)

#### Memory Systems
- **Crystalline lattice memory** - 3D memory structure
- **FAISS-like vector indexing** for semantic memory
- **Session persistence** - Long-term memory logging
- **Emotional context encoding** - Memories stored with emotional context

### 6. **Real-Time Audio Processing**

#### Audio System
- **Rust bio-acoustic engine** (`libbio_audio.so`) for performance-critical processing
- **Neural TTS** (Coqui TTS) for voice synthesis
- **Real-time audio bridge** for low-latency processing
- **Device selection** and audio routing

#### Voice Features
- **Prosody extraction** - Analyzes pitch, energy, rhythm
- **Prosody transfer** - Maintains user's voice characteristics in corrected speech
- **Voice adaptation** - Lifelong learning of user's voice patterns

### 7. **Therapeutic Gaming & Visualization**

#### Cognitive Nebula
- **3D universe visualization** using React + Three.js
- **AI image generation** from voice input
- **Therapeutic gaming mechanics**
- **Real-time voice bubble visualization** - 3D bubble that responds to voice energy, pitch, and speech sounds

#### Voice Bubble
- **Real-time 3D visualization** of voice
- **Bouba/Kiki effect** - Shape morphing based on speech sounds
- **Energy-reactive particles**
- **Volumetric light rays**
- **Shader-based deformation**

### 8. **Production Features**

#### Deployment Capabilities
- **Web API interface** (Flask) for external integration
- **GUI interface** (PySide6/Kivy) for user interaction
- **Session management** and persistence
- **User profiles** and configuration
- **Production logging** and monitoring
- **Safety monitoring** and emergency controls

#### Configuration
- **Voice profile management** - Store and load voice samples
- **Therapeutic settings** - Customize interventions
- **Audio device selection**
- **Model path configuration**

## 🔄 How It Works

### Typical User Flow

1. **User speaks** into microphone
2. **System captures** audio with autism-optimized VAD (respects pauses)
3. **Speech transcribed** to text
4. **Linguistic correction** applied (grammar, pronunciation)
5. **Emotional state analyzed** via Crystalline Heart
6. **ABA intervention** triggered if needed (calming, reinforcement, etc.)
7. **Corrected speech synthesized** in user's cloned voice
8. **Prosody transferred** to maintain natural voice characteristics
9. **Audio played back** to user
10. **Visual feedback** shown in 3D Cognitive Nebula
11. **Memory encoded** with emotional context
12. **Progress tracked** for therapeutic goals

### Therapeutic Benefits

- **Auditory Prediction Error (APE) therapy** - Users hear their corrected speech in their own voice
- **Real-time feedback** - Immediate correction without judgment
- **Voice confidence** - Builds speaking confidence through positive reinforcement
- **Emotional regulation** - Helps manage anxiety, overwhelm, and sensory issues
- **Communication skills** - Improves social communication through practice
- **Self-awareness** - Visual feedback helps understand speech patterns

## 🎯 Use Cases

1. **Autism Speech Therapy** - Primary use case for speech and communication support
2. **Voice Therapy** - General speech correction and improvement
3. **Language Learning** - Pronunciation and grammar practice
4. **Therapeutic Gaming** - Engaging visual feedback for therapy
5. **Research Platform** - AGI and consciousness research
6. **Accessibility Tool** - Communication support for various needs

## 🏗️ Technical Architecture

### Components
- **Python backend** - Main system logic
- **Rust core** - Performance-critical audio processing
- **React frontend** - Cognitive Nebula visualization
- **Neural TTS** - Voice cloning and synthesis
- **Quantum systems** - Advanced computation
- **Memory systems** - Long-term learning

### Key Technologies
- NumPy, SciPy - Scientific computing
- PyTorch/TensorFlow - Neural networks
- Coqui TTS - Voice synthesis
- Vosk/Sherpa-ONNX - Speech recognition
- Three.js - 3D visualization
- React - Frontend framework
- Rust - Performance core

## 📊 System Metrics

The system tracks:
- **Global Coherence Level (GCL)** - Emotional stability
- **Stress levels** - System and user stress
- **Life intensity** - System vitality metric
- **ABA success rates** - Therapeutic progress
- **Voice adaptation metrics** - Voice learning progress
- **Memory stability** - Memory system health
- **Processing latency** - Real-time performance

---

**In Summary**: This is a **comprehensive therapeutic AGI system** that provides **real-time voice therapy** through **voice cloning**, **speech correction**, **emotional regulation**, and **therapeutic interventions**, specifically optimized for **autistic individuals** but useful for anyone needing speech therapy or communication support.

