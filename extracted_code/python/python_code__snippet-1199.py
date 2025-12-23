class BubbleState:
    radii: np.ndarray  # [N_vertices]
    colors: np.ndarray  # [N_vertices, 3]
    pbr_props: Dict[str, float]  # {"rough", "metal", "spike"}


def compute_bubble_state(
    vertices: np.ndarray,
    profile: VoiceProfile,
    attempt_feat: AttemptFeatures,
    t_idx: int,
    layout: Optional[Dict[str, Any]] = None,
    *,
    base_radius: Optional[float] = None,
