from voice_field import generate_voice_field
from bubble_foam import compute_bubble_state_vertices

for t_idx in range(len(controls['energy'])):
    f0 = controls['f0'][t_idx]
    field = generate_voice_field(vertices, f0, t=t_idx*0.01)
    
    state = compute_bubble_state_vertices(
        vertices, profile, attempt, t_idx,
        layout={"voice_field": field}
    )
    
    render_mesh(state.radii, state.colors, state.pbr_props)
