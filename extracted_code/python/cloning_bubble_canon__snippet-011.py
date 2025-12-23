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
