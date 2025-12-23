from bubble_synthesizer import controls_to_attempt_features
from bubble_foam import compute_bubble_state

attempt = controls_to_attempt_features(controls, dt=0.01)

for t in range(len(controls['energy'])):
    state = compute_bubble_state(profile, attempt, t_time=t*0.01)
    render_bubble(state['radius'], state['spike'])
