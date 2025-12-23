from bubble_foam import compute_bubble_state

state = compute_bubble_state(profile, attempt_feat, t_time=1.0)
# Returns: {"radius": 1.2, "color_r": 0.8, "color_g": 0.3, ...}
