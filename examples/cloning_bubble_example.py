"""
Complete end-to-end example of the Cloning Bubble system.

Demonstrates:
1. Enrollment (creating the bubble from child audio)
2. Synthesis (text → child voice + bubble controls)
3. Visualization (bubble state computation)

Usage:
    python example_cloning_bubble.py
"""

import numpy as np
from pathlib import Path

# Import the Cloning Bubble modules
from goeckoh.psychoacoustic_engine.voice_logger import log_voice_characteristics, load_speaker_profile
from goeckoh.psychoacoustic_engine.bubble_synthesizer import feed_text_through_bubble, controls_to_attempt_features
from goeckoh.psychoacoustic_engine.bubble_foam import compute_bubble_state
from goeckoh.psychoacoustic_engine.voice_field import generate_voice_field


def create_mock_audio(duration: float = 2.0, sr: int = 22050) -> np.ndarray:
    """Generate mock audio for demonstration."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate speech with varying pitch and amplitude
    f0 = 200 + 30 * np.sin(2 * np.pi * t * 2)  # Pitch variation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * t * 3)  # Amplitude modulation
    
    audio = envelope * np.sin(2 * np.pi * f0 * t)
    
    # Add some noise for realism
    audio += 0.05 * np.random.randn(len(audio))
    
    return audio.astype(np.float32)


def create_mock_embedding(dim: int = 256) -> np.ndarray:
    """Generate mock speaker embedding (in production, use OpenVoice/SpeechBrain)."""
    return np.random.randn(dim).astype(np.float32)


def create_sphere_vertices(n_lat: int = 20, n_lon: int = 20) -> np.ndarray:
    """Generate sphere vertices for bubble mesh."""
    vertices = []
    
    for i in range(n_lat):
        theta = np.pi * i / (n_lat - 1)
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            vertices.append([x, y, z])
    
    return np.array(vertices, dtype=np.float32)


def main():
    """Complete Cloning Bubble workflow."""
    
    print("=" * 60)
    print("CLONING BUBBLE - Psychoacoustic Voice Identity System")
    print("=" * 60)
    
    # Configuration
    sr = 22050
    user_id = "alice"
    output_dir = "./bubble_data"
    
    # ========================================
    # PHASE 1: ENROLLMENT
    # ========================================
    print("\n[PHASE 1: ENROLLMENT]")
    print("Creating Cloning Bubble from child audio samples...")
    
    # In production: load real audio files
    # audio_samples = [librosa.load(f, sr=sr)[0] for f in audio_files]
    audio_samples = [
        create_mock_audio(duration=2.0, sr=sr),
        create_mock_audio(duration=1.5, sr=sr),
        create_mock_audio(duration=2.5, sr=sr),
    ]
    
    # In production: use real voice encoder (OpenVoice, SpeechBrain, etc.)
    # embedding = voice_encoder.encode(audio_samples)
    speaker_embedding = create_mock_embedding(dim=256)
    
    # Create the Cloning Bubble (Θᵤ + embedding)
    profile = log_voice_characteristics(
        audio_samples=audio_samples,
        sr=sr,
        user_id=user_id,
        output_dir=output_dir,
        speaker_embedding=speaker_embedding,
    )
    
    print(f"\n✓ Bubble Identity Created:")
    print(f"  User: {profile.user_id}")
    print(f"  Embedding shape: {profile.embedding.shape}")
    
    # ========================================
    # PHASE 2: SYNTHESIS
    # ========================================
    print("\n[PHASE 2: SYNTHESIS]")
    print("Transforming text through the Cloning Bubble...")
    
    text = "Hello, this is my voice in the bubble!"
    
    # Generate child-shaped speech and bubble controls
    audio, controls = feed_text_through_bubble(
        text=text,
        profile=profile,
        vocoder_backend=None,  # Uses MockVocoder
        dt=0.01  # 10ms frames
    )
    
    print(f"\n✓ Synthesis Complete:")
    print(f"  Text: '{text}'")
    print(f"  Audio: {audio.shape[0]} samples ({audio.shape[0]/sr:.2f}s)")
    print(f"  Controls: {len(controls['energy'])} frames")
    print(f"  Frame rate: {1/controls['dt'][0]:.1f} Hz")
    
    # ========================================
    # PHASE 3: VISUALIZATION
    # ========================================
    print("\n[PHASE 3: VISUALIZATION]")
    print("Computing bubble states for animation...")
    
    # Create bubble mesh
    vertices = create_sphere_vertices(n_lat=20, n_lon=20)
    print(f"  Bubble vertices: {vertices.shape[0]}")
    
    # Convert controls to AttemptFeatures
    attempt = controls_to_attempt_features(controls)
    
    # Compute bubble states for each frame
    num_frames = len(controls['energy'])
    bubble_states = []
    
    for t_idx in range(num_frames):
        # Optionally generate voice field for this frame
        f0 = float(controls['f0'][t_idx])
        t = t_idx * controls['dt'][0]
        voice_field = generate_voice_field(vertices, f0, t, num_modes=3)
        
        # Compute bubble state
        state = compute_bubble_state(
            vertices=vertices,
            profile=profile,
            attempt_feat=attempt,
            t_idx=t_idx,
            layout={"voice_field": voice_field},
            base_radius=1.0
        )
        
        bubble_states.append(state)
    
    print(f"\n✓ Visualization Ready:")
    print(f"  Frames: {len(bubble_states)}")
    
    # Show sample state
    sample_state = bubble_states[num_frames // 2]
    print(f"\n  Sample Frame (t={num_frames//2}):")
    print(f"    Radius range: [{sample_state.radii.min():.3f}, {sample_state.radii.max():.3f}]")
    print(f"    Color: RGB{tuple(sample_state.colors[0])}")
    print(f"    Roughness: {sample_state.pbr_props['rough']:.3f}")
    print(f"    Metalness: {sample_state.pbr_props['metal']:.3f}")
    print(f"    Spikiness: {sample_state.pbr_props['spike']:.3f}")
    
    # ========================================
    # SHADER INTEGRATION
    # ========================================
    print("\n[SHADER INTEGRATION]")
    print("To render the bubble, send these uniforms to your shader:")
    print(f"""
    // Per-frame uniforms
    uniform float uTime = {t:.3f};
    uniform float uRoughness = {sample_state.pbr_props['rough']:.3f};
    uniform float uMetalness = {sample_state.pbr_props['metal']:.3f};
    uniform float uSpikeAmount = {sample_state.pbr_props['spike']:.3f};
    
    // Per-vertex attributes (from sample_state.radii)
    attribute float aRadius;  // Per-vertex displacement
    
    // In vertex shader:
    vec3 newPos = position + normal * aRadius;
    """)
    
    # ========================================
    # RELOAD TEST
    # ========================================
    print("\n[RELOAD TEST]")
    print("Testing profile persistence...")
    
    loaded_profile = load_speaker_profile(user_id, output_dir)
    
    print(f"\n✓ Profile Reloaded:")
    print(f"  Pitch matches: {np.isclose(loaded_profile.fingerprint.mu_f0, profile.fingerprint.mu_f0)}")
    print(f"  Embedding matches: {np.allclose(loaded_profile.embedding, profile.embedding)}")
    
    print("\n" + "=" * 60)
    print("Cloning Bubble system fully operational!")
    print("=" * 60)
    
    return profile, audio, controls, bubble_states


if __name__ == "__main__":
    profile, audio, controls, states = main()
    
    print("\n[NEXT STEPS]")
    print("1. Replace MockVocoder with real TTS (Coqui, OpenVoice, XTTS)")
    print("2. Integrate with Three.js/WebGL for real-time rendering")
    print("3. Connect to audio playback synchronized with bubble animation")
    print("4. Add error correction UI for speech practice")
