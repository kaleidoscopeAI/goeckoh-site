# Cloning Bubble – Psychoacoustic Voice Bubble (Canon)

This document formalizes the **Cloning Bubble**: a closed-loop psychoacoustic system that:

1. Listens to a child and logs their **Bubble DNA** (Θᵤ).
2. Uses Θᵤ to constrain both **speech synthesis** and **visual bubble dynamics**.
3. Produces corrected first-person speech that **sounds and looks like the child**.

All code paths referenced here live under `goeckoh/psychoacoustic_engine/`.

---

## 1. Core Concepts

### 1.1 Bubble DNA (Θᵤ)

For each child `u` we store a `VoiceFingerprint`:

- **μ_F0** – median pitch (voice color)
- **σ_F0** – pitch variability (expressivity range)
- **base_roughness** – from HNR (breathy vs clean)
- **base_metalness** – from spectral tilt (soft vs bright)
- **base_sharpness** – from ZCR (Bouba vs Kiki baseline)
- **Rateᵤ** – syllables per second (idle heartbeat / tempo)
- **jitter_base, shimmer_base** – micro-variations for realism
- **base_radius** – default bubble size

We also keep a **neural embedding**:

- **embedding** – tone-color vector from an open voice encoder (e.g., OpenVoice / SpeechBrain).

Together:

> **SpeakerProfile = (Θᵤ, embedding)**

This is the **Cloning Bubble** identity.

---

## 2. Mathematical Foundation

### 2.1 Bubble Radius with Bouba/Kiki Texture

For vertex `n` at time `t`:

```
R(n, t) = R₀ * [ 1 + u_energy(t) + β ∑_k u_k(n,t) + χ(t) * γ_spike * |N_spiky(n)| ]
```

Where:

- `R₀` – base radius from fingerprint
- `u_energy(t)` – normalized RMS energy [0, 1.5]
- `u_k(n,t)` – modal displacement field (voice harmonics)
- `χ(t) = smoothstep(Z(t), 0.10, 0.40)` – Bouba/Kiki coefficient from ZCR
- `γ_spike = 0.12` – spike gain
- `N_spiky(n)` – deterministic spatial noise at vertex `n`

### 2.2 Idle Heartbeat

When no active speech (energy below threshold):

**Activity Gate:**
```
G_active(t) = sigmoid((Volume(t) - 0.05) * 10.0)
```

**Idle Frequency:**
```
ω_idle = 2π * Rateᵤ
```

**Blended Radius:**
```
R_total(n,t) = G_active(t) * R_active(n,t) + (1 - G_active(t)) * R_idle(t)
```

Where:
```
R_idle(t) = R₀ * (0.85 + A_idle * sin(ω_idle * t))
```
with `A_idle = 0.05`.

### 2.3 Material State (PBR)

```
Roughness(t) = 1.0 - HNR(t)
Metalness(t) = Tilt(t) * 1.5  [clamped to 0..1]
Spike(t) = χ(t)  [from ZCR]
```

### 2.4 Color from Pitch

```
Hue = (μ_F0 - 80) / (400 - 80)  [maps 80-400 Hz to 0-1]
RGB = hue_to_rgb(Hue)  [HSV conversion with S=1, V=1]
```

---

## 3. Data Structures

### 3.1 Voice Profile

```python
@dataclass
class VoiceFingerprint:
    """Static Bubble Constraints Θ_u for one child."""
    mu_f0: float
    sigma_f0: float
    base_roughness: float    # 0..1
    base_metalness: float    # 0..1
    base_sharpness: float    # 0..1
    rate: float              # syllables/sec
    jitter_base: float
    shimmer_base: float
    base_radius: float = 1.0


@dataclass
class SpeakerProfile:
    user_id: str
    fingerprint: VoiceFingerprint
    embedding: np.ndarray    # 1D float32 vector
```

### 3.2 Attempt Features (Per-Utterance Psychoacoustics)

```python
@dataclass
class AttemptFeatures:
    energy_attempt: np.ndarray       # [T] - RMS energy
    f0_attempt: np.ndarray           # [T] - fundamental frequency
    zcr_attempt: np.ndarray          # [T] - zero-crossing rate
    spectral_tilt: np.ndarray        # [T] - spectral slope
    hnr_attempt: np.ndarray          # [T] - harmonics-to-noise ratio
    dt: float                        # seconds per frame
```

`AttemptFeatures` is used both for **real speech analysis** and for **synthesized control curves**.

### 3.3 Bubble State (Visual Output)

```python
@dataclass
class BubbleState:
    radii: np.ndarray          # [N_vertices]
    colors: np.ndarray         # [N_vertices, 3] RGB
    pbr_props: Dict[str, float]  # {"rough", "metal", "spike"}
```

---

## 4. Enrollment – Creating the Cloning Bubble

### 4.1 Logger Overview

The **Logger** listens to several child audio clips and produces:

- A **VoiceFingerprint (Θᵤ)** from acoustic analysis
- A **tone-color embedding** from an open encoder

### 4.2 Enrollment Process

```python
from goeckoh.psychoacoustic_engine.voice_logger import log_voice_characteristics

profile = log_voice_characteristics(
    audio_samples=[y1, y2, y3],  # List of numpy arrays
    sr=22050,
    user_id="alice",
    output_dir="./bubble_data",
    speaker_embedding=encoder.encode(audio)  # From OpenVoice/SpeechBrain
)
```

**Saves:**
- `{user_id}_fingerprint.json` – Θᵤ parameters
- `{user_id}_embed.npy` – Neural embedding

**Result:** A `SpeakerProfile` that *is* the Cloning Bubble for that child.

---

## 5. Text → Bubble → Child Voice

### 5.1 Synthesis Pipeline

```python
from goeckoh.psychoacoustic_engine.bubble_synthesizer import feed_text_through_bubble

audio, controls = feed_text_through_bubble(
    text="Hello world",
    profile=profile,
    vocoder_backend=my_vocoder,  # Real TTS backend
    dt=0.01  # 10ms frames
)
```

**Returns:**
- `audio` – Synthesized speech with child's timbre
- `controls` – Dict with `{energy, f0, zcr, hnr, tilt, dt}` aligned to frames

### 5.2 Phoneme Sharpness Mapping

The synthesizer maps phoneme classes to ZCR targets:

- **Vowels** → ZCR = 0.1 (Bouba / smooth)
- **Sharp consonants** (K, T, P, S, ...) → ZCR = 0.9 (Kiki / spiky)
- **Soft consonants** (M, N, L, R, ...) → ZCR = 0.5 (intermediate)

Blends with child's `base_sharpness` to create frame-by-frame ZCR contour.

### 5.3 Control Curves

All synthesis parameters are derived from Θᵤ:

**Pitch contour:**
```python
target_f0 = μ_F0 + prosody(t) * σ_F0 + jitter(t)
```

**Energy contour:**
```python
energy = base_prosody(t) + shimmer(t)
```

**Material contours:**
```python
hnr = 1.0 - base_roughness  (constant)
tilt = base_metalness       (constant)
```

**Bouba/Kiki contour:**
```python
zcr[i] = blend(phoneme_target[i], base_sharpness, α=0.6)
```

---

## 6. Bubble Visualization

### 6.1 State Computation

```python
from goeckoh.psychoacoustic_engine.bubble_foam import compute_bubble_state
from goeckoh.psychoacoustic_engine.bubble_synthesizer import controls_to_attempt_features

# Convert synthesis controls to features
attempt = controls_to_attempt_features(controls)

# Generate per-vertex voice field (optional)
from goeckoh.psychoacoustic_engine.voice_field import generate_voice_field
voice_field = generate_voice_field(vertices, f0=controls['f0'][t_idx], t=t)

# Compute bubble state for frame t_idx
state = compute_bubble_state(
    vertices=vertices,           # [N, 3] mesh positions
    profile=profile,             # Child's identity
    attempt_feat=attempt,        # Acoustic features
    t_idx=t_idx,                 # Current frame
    layout={"voice_field": voice_field},  # Optional pre-computed field
    base_radius=1.0
)
```

**Output:**
- `state.radii` – Per-vertex displacement `[N]`
- `state.colors` – Per-vertex RGB `[N, 3]`
- `state.pbr_props` – Uniform material properties `{"rough", "metal", "spike"}`

### 6.2 Modal Voice Field

For spatial coherence, vertices can have deterministic modal waves:

```python
u_k(n,t) = ∑_{k=1}^K (decay^(k-1) / k) * sin(k * ω₀ * t + φ_k(n))
```

Where `φ_k(n)` is a deterministic phase from vertex position:

```python
φ_k(n) = hash(position[n], mode_idx=k) * 2π
```

This creates spatially-varying but reproducible patterns on the bubble surface.

---

## 7. Shader Integration (GLSL)

### 7.1 Vertex Shader

```glsl
uniform float uTime;
uniform float uSpikeAmount;  // from pbr_props["spike"]
uniform float uRoughness;    // from pbr_props["rough"]
uniform float uMetalness;    // from pbr_props["metal"]

attribute vec3 position;
attribute vec3 normal;
attribute float aRadius;     // from state.radii[n]

varying float vRoughness;
varying float vMetalness;

float hash3d(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

void main() {
    // Deterministic Kiki spikes
    float kiki = uSpikeAmount * hash3d(position * 10.0);
    
    // Idle jitter
    float jitter = 0.5 * sin(uTime * 13.0) + 0.5 * sin(uTime * 23.0);
    
    // Total displacement
    float offset = aRadius + 0.1 * jitter + kiki;
    
    vec3 newPos = position + normal * offset;
    
    vRoughness = uRoughness;
    vMetalness = uMetalness;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
}
```

### 7.2 Fragment Shader (PBR)

```glsl
varying float vRoughness;
varying float vMetalness;
varying vec3 vNormal;

void main() {
    // Simplified PBR lighting
    vec3 baseColor = vec3(1.0, 0.5, 0.3);  // From state.colors
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    
    float NdotL = max(dot(vNormal, lightDir), 0.0);
    vec3 diffuse = baseColor * NdotL * (1.0 - vMetalness);
    
    vec3 specular = vec3(vMetalness) * pow(NdotL, 1.0 / (vRoughness + 0.01));
    
    gl_FragColor = vec4(diffuse + specular, 1.0);
}
```

---

## 8. End-to-End Flow

### Complete Pipeline

1. **Enrollment:**
   ```python
   profile = log_voice_characteristics(audio_clips, sr, user_id, out_dir, embedding)
   ```

2. **Synthesis:**
   ```python
   audio, controls = feed_text_through_bubble(text, profile, vocoder, dt)
   ```

3. **Visualization:**
   ```python
   attempt = controls_to_attempt_features(controls)
   for t in range(num_frames):
       state = compute_bubble_state(vertices, profile, attempt, t)
       render_bubble(state)  # Send to GPU
   ```

4. **Playback:**
   - Play `audio` synchronized with bubble animation
   - Each frame's `state` drives shader uniforms and vertex attributes

---

## 9. Properties

### 9.1 Determinism

The system is **fully deterministic**:

- No random number generators at runtime
- Same text + profile → identical audio + bubble
- Replay-safe for debugging and version control
- Spatial noise uses deterministic hash functions

### 9.2 Child-Specificity

Every aspect respects the child's Θᵤ:

- **Pitch range:** [μ_F0 - σ_F0, μ_F0 + σ_F0]
- **Speaking rate:** Rateᵤ syllables/second
- **Voice quality:** roughness, metalness, sharpness
- **Bubble color:** Derived from μ_F0
- **Idle heartbeat:** Frequency = Rateᵤ

### 9.3 Audio-Visual Lock

The same `AttemptFeatures` drive both:

- **Vocoder:** Synthesis with child's embedding + controls
- **Bubble:** Geometric and material state

This ensures **perfect synchronization** between sound and visuals.

---

## 10. Integration Points

### 10.1 Real TTS Backend

Replace `MockVocoder` with:

- **Coqui TTS** (open source, multi-speaker)
- **OpenVoice** (tone color cloning)
- **XTTS** (cross-lingual synthesis)
- **Bark** (expressive speech)

The vocoder must accept:
- Speaker embedding
- Control curves (F0, energy, HNR, tilt)
- Phoneme sequence

### 10.2 Voice Encoder

For `speaker_embedding`, use:

- **OpenVoice encoder** (recommended)
- **SpeechBrain speaker recognition**
- **Resemblyzer**
- **TitaNet**

### 10.3 Frontend (Three.js)

```javascript
// Load bubble mesh
const geometry = new THREE.IcosahedronGeometry(1.0, 3);

// Per-frame update
function animate(frameIdx) {
    const state = bubbleStates[frameIdx];
    
    // Update vertex positions
    for (let i = 0; i < vertices.length; i++) {
        const pos = baseVertices[i].clone();
        const normal = normals[i];
        pos.add(normal.multiplyScalar(state.radii[i]));
        geometry.attributes.position.array[i*3] = pos.x;
        geometry.attributes.position.array[i*3+1] = pos.y;
        geometry.attributes.position.array[i*3+2] = pos.z;
    }
    geometry.attributes.position.needsUpdate = true;
    
    // Update material uniforms
    material.uniforms.uRoughness.value = state.pbr_props.rough;
    material.uniforms.uMetalness.value = state.pbr_props.metal;
    material.uniforms.uSpikeAmount.value = state.pbr_props.spike;
}
```

---

## 11. Future Extensions

### 11.1 Multi-Child Interpolation

Blend between multiple children's profiles:

```python
blended_profile = interpolate_profiles([profile_a, profile_b], weight=0.5)
```

### 11.2 Emotion Control

Add emotional contours on top of base Θᵤ:

```python
audio, controls = feed_text_through_bubble(
    text, profile, vocoder, dt,
    emotion={"happy": 0.8, "excited": 0.5}
)
```

### 11.3 Real-Time Correction

Compare child's attempt with synthesized target:

```python
real_features = analyze_attempt(child_audio, sr)
synth_features = controls_to_attempt_features(controls)
error = compute_psychoacoustic_distance(real_features, synth_features)
```

---

## 12. Credits

This Cloning Bubble architecture transforms text-to-speech from a generic tool into a **psychoacoustic mold** that forces any text to take the child's voice shape—in both sound and visual form.

The system is:
- **Deterministic** (no runtime randomness)
- **Child-specific** (every parameter from Θᵤ)
- **Audio-visual locked** (same features drive sound and bubble)
- **Modular** (swap vocoder, encoder, renderer)
- **Scientifically grounded** (psychoacoustic features, not arbitrary mappings)

---

**End of Canon**
