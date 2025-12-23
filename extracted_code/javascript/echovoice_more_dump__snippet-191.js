let node_features = vec![
    features.element_symbol.clone(),
    features.electronegativity,
    features.valence_electrons,
    features.atomic_radius,
    features.is_metal as u8 as f32,
    features.mass,
    features.electron_affinity,
