"""
Test runner for the Cloning Bubble psychoacoustic core.

Run this to verify all tests pass:
    python test_runner.py
"""

import sys
import numpy as np

sys.path.insert(0, '.')


# Now run the actual tests
def test_analyze_chunk_silence():
    """Ensure silence returns near-zero energy and valid shapes."""
    from cloning_bubble.core.attempt_analysis import analyze_chunk
    
    mock_audio_silence = np.zeros(22050, dtype=np.float32)
    feat = analyze_chunk(mock_audio_silence, sr=22050)
    
    assert feat.energy_attempt[0] < 0.001, f"Energy should be near-zero, got {feat.energy_attempt[0]}"
    assert feat.f0_attempt[0] == 0.0, f"F0 should be 0 for silence, got {feat.f0_attempt[0]}"
    print("✓ test_analyze_chunk_silence passed")


def test_analyze_chunk_sine_wave():
    """Ensure a pure sine wave is detected correctly."""
    from cloning_bubble.core.attempt_analysis import analyze_chunk
    
    sr = 22050
    t = np.linspace(0, 1, sr)
    mock_audio_sine = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    
    feat = analyze_chunk(mock_audio_sine, sr=22050)
    
    # Energy should be approx 0.5 / sqrt(2) ~= 0.35
    energy = feat.energy_attempt[0]
    assert 0.3 < energy < 0.4, f"Energy should be ~0.35, got {energy}"
    
    # Pitch should be approx 440Hz
    f0 = feat.f0_attempt[0]
    assert 400 < f0 < 480, f"F0 should be ~440Hz, got {f0}"
    print("✓ test_analyze_chunk_sine_wave passed")


def test_compute_bubble_state_idle():
    """Test bubble state when audio energy is zero (Idle Mode)."""
    from cloning_bubble.core.voice_profile import VoiceFingerprint, SpeakerProfile
    from cloning_bubble.core.attempt_analysis import AttemptFeatures
    from cloning_bubble.core.bubble_foam import compute_bubble_state
    
    fp = VoiceFingerprint(
        mu_f0=200.0,
        sigma_f0=20.0,
        base_roughness=0.1,
        base_metalness=0.5,
        base_sharpness=0.2,
        rate=4.0,
        jitter_base=0.05,
        shimmer_base=0.05
    )
    default_profile = SpeakerProfile(user_id="test_user", fingerprint=fp)
    
    # Create a dummy feature frame with 0 energy
    feat = AttemptFeatures(
        energy_attempt=np.array([0.0]),
        f0_attempt=np.array([0.0]),
        zcr_attempt=np.array([0.0]),
        spectral_tilt=np.array([0.0]),
        hnr_attempt=np.array([0.0]),
        dt=0.01
    )
    
    state = compute_bubble_state(default_profile, feat, t_time=1.0)
    
    # Radius should be near base_radius (1.0) * breathing factor
    # Idle breathing is 0.9 + 0.1*sin(...)
    radius = state['radius']
    assert 0.8 <= radius <= 1.0, f"Idle radius should be 0.8-1.0, got {radius}"
    assert state['spike'] == 0.0, f"No spikes in idle, got {state['spike']}"
    print("✓ test_compute_bubble_state_idle passed")


def test_compute_bubble_state_active():
    """Test bubble expansion under high energy."""
    from cloning_bubble.core.voice_profile import VoiceFingerprint, SpeakerProfile
    from cloning_bubble.core.attempt_analysis import AttemptFeatures
    from cloning_bubble.core.bubble_foam import compute_bubble_state
    
    fp = VoiceFingerprint(
        mu_f0=200.0,
        sigma_f0=20.0,
        base_roughness=0.1,
        base_metalness=0.5,
        base_sharpness=0.2,
        rate=4.0,
        jitter_base=0.05,
        shimmer_base=0.05
    )
    default_profile = SpeakerProfile(user_id="test_user", fingerprint=fp)
    
    feat = AttemptFeatures(
        energy_attempt=np.array([0.8]), # Loud
        f0_attempt=np.array([200.0]),
        zcr_attempt=np.array([0.5]),    # Spiky
        spectral_tilt=np.array([0.5]),
        hnr_attempt=np.array([0.9]),
        dt=0.01
    )
    
    state = compute_bubble_state(default_profile, feat, t_time=1.0)
    
    # Radius should expand significantly
    # r_active = 1.0 * (1.0 + 0.8 * 2.0) = 2.6
    # It blends with idle, but should be > 1.5
    radius = state['radius']
    assert radius > 1.5, f"Active radius should be > 1.5, got {radius}"
    assert state['spike'] > 0.0, f"Should have spikes, got {state['spike']}"
    print("✓ test_compute_bubble_state_active passed")


def test_feed_text_through_bubble():
    """Ensure synthesizer produces audio and aligned control curves."""
    from cloning_bubble.core.voice_profile import VoiceFingerprint, SpeakerProfile
    from cloning_bubble.core.bubble_synthesizer import feed_text_through_bubble
    
    fp = VoiceFingerprint(
        mu_f0=200.0,
        sigma_f0=20.0,
        base_roughness=0.1,
        base_metalness=0.5,
        base_sharpness=0.2,
        rate=4.0,
        jitter_base=0.05,
        shimmer_base=0.05
    )
    default_profile = SpeakerProfile(user_id="test_user", fingerprint=fp)
    
    text = "Hello"
    dt = 0.01
    audio, controls = feed_text_through_bubble(text, default_profile, dt=dt)
    
    # Check Audio
    assert isinstance(audio, np.ndarray), "Audio should be ndarray"
    assert len(audio) > 0, "Audio should have samples"
    
    # Check Controls
    expected_keys = {'energy', 'f0', 'zcr', 'tilt', 'hnr'}
    assert expected_keys.issubset(controls.keys()), f"Missing keys: {expected_keys - set(controls.keys())}"
    
    # Check Alignment
    n_frames = len(controls['energy'])
    assert len(controls['f0']) == n_frames, "F0 length should match energy"
    
    print("✓ test_feed_text_through_bubble passed")


def test_cloning_bubble_core_shim_api():
    """Verify cloning_bubble.core shim exposes the expected public API."""
    # Import from the shim package
    from cloning_bubble.core import (
        AttemptFeatures,
        analyze_chunk,
        BubbleState,
        compute_bubble_state,
        MockVocoder,
        feed_text_through_bubble,
        SpeakerProfile,
        VoiceFingerprint,
    )
    
    # Import from the underlying modules
    import attempt_analysis
    import bubble_foam
    import bubble_synthesizer
    import voice_profile
    
    # Verify that the shim exposes the correct objects
    assert AttemptFeatures is attempt_analysis.AttemptFeatures, \
        "cloning_bubble.core.AttemptFeatures should reference attempt_analysis.AttemptFeatures"
    assert analyze_chunk is attempt_analysis.analyze_chunk, \
        "cloning_bubble.core.analyze_chunk should reference attempt_analysis.analyze_chunk"
    assert BubbleState is bubble_foam.BubbleState, \
        "cloning_bubble.core.BubbleState should reference bubble_foam.BubbleState"
    assert compute_bubble_state is bubble_foam.compute_bubble_state, \
        "cloning_bubble.core.compute_bubble_state should reference bubble_foam.compute_bubble_state"
    assert MockVocoder is bubble_synthesizer.MockVocoder, \
        "cloning_bubble.core.MockVocoder should reference bubble_synthesizer.MockVocoder"
    assert feed_text_through_bubble is bubble_synthesizer.feed_text_through_bubble, \
        "cloning_bubble.core.feed_text_through_bubble should reference bubble_synthesizer.feed_text_through_bubble"
    assert SpeakerProfile is voice_profile.SpeakerProfile, \
        "cloning_bubble.core.SpeakerProfile should reference voice_profile.SpeakerProfile"
    assert VoiceFingerprint is voice_profile.VoiceFingerprint, \
        "cloning_bubble.core.VoiceFingerprint should reference voice_profile.VoiceFingerprint"
    
    print("✓ test_cloning_bubble_core_shim_api passed")


def test_analyze_chunk_energy_scales_with_amplitude():
    """Verify RMS energy scales with amplitude (no peak normalization)."""
    from cloning_bubble.core.attempt_analysis import analyze_chunk
    
    # 1 second of 1 kHz sine wave at amplitude 0.5
    sr = 16000
    t = np.linspace(0, 1.0, int(sr), endpoint=False, dtype=np.float32)
    y1 = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y2 = 2.0 * y1  # double the amplitude of y1
    
    # analyze_chunk should now operate on raw float32 without peak normalization
    features1 = analyze_chunk(y1, sr)
    features2 = analyze_chunk(y2, sr)
    
    # The energy feature is called "energy_attempt" in AttemptFeatures
    rms1 = features1.energy_attempt[0]
    rms2 = features2.energy_attempt[0]
    
    # RMS should scale approximately linearly with amplitude
    assert rms2 > rms1, f"RMS of doubled signal should be greater: {rms2} > {rms1}"
    np.testing.assert_allclose(rms2 / rms1, 2.0, rtol=0.2, 
                               err_msg=f"RMS should scale linearly: {rms2}/{rms1} ≈ 2.0")
    
    print("✓ test_analyze_chunk_energy_scales_with_amplitude passed")


def test_analyze_chunk_energy_differs_for_different_peak_amplitudes():
    """Verify energy differs for signals with different peak amplitudes."""
    from cloning_bubble.core.attempt_analysis import analyze_chunk
    
    sr = 16000
    t = np.linspace(0, 1.0, int(sr), endpoint=False, dtype=np.float32)
    
    # Two signals with identical shape but different amplitudes
    base = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    quiet = 0.25 * base
    loud = 0.75 * base
    
    features_quiet = analyze_chunk(quiet, sr)
    features_loud = analyze_chunk(loud, sr)
    
    rms_quiet = features_quiet.energy_attempt[0]
    rms_loud = features_loud.energy_attempt[0]
    
    # Without peak normalization, louder signal must have higher energy
    assert rms_loud > rms_quiet, \
        f"Louder signal must have higher energy: {rms_loud} > {rms_quiet}"
    
    # Verify the ratio is approximately 3:1 (0.75 / 0.25)
    np.testing.assert_allclose(rms_loud / rms_quiet, 3.0, rtol=0.2,
                               err_msg=f"Energy ratio should be ~3.0: {rms_loud}/{rms_quiet}")
    
    print("✓ test_analyze_chunk_energy_differs_for_different_peak_amplitudes passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CLONING BUBBLE - Psychoacoustic Core Test Suite")
    print("=" * 60)
    
    tests = [
        ("Analyze Chunk - Silence", test_analyze_chunk_silence),
        ("Analyze Chunk - Sine Wave", test_analyze_chunk_sine_wave),
        ("Bubble State - Idle Mode", test_compute_bubble_state_idle),
        ("Bubble State - Active Mode", test_compute_bubble_state_active),
        ("Text Synthesis", test_feed_text_through_bubble),
        ("Cloning Bubble Core Shim API", test_cloning_bubble_core_shim_api),
        ("Energy Scales with Amplitude", test_analyze_chunk_energy_scales_with_amplitude),
        ("Energy Differs for Different Peak Amplitudes", test_analyze_chunk_energy_differs_for_different_peak_amplitudes),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! The Cloning Bubble is operational.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
