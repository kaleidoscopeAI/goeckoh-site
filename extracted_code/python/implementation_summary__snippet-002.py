from bubble_foam import compute_bubble_state_vertices

state = compute_bubble_state_vertices(
    vertices, profile, attempt_feat, t_idx=10,
    layout={"voice_field": field}
)
# Returns: BubbleState(radii=[N], colors=[N,3], pbr_props={...})
