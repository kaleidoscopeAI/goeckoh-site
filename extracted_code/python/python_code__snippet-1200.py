    # Bouba/Kiki spike coefficient χ(t) from ZCR
    zcr = float(attempt_feat.zcr_attempt[t_idx])
    chi = _smoothstep(zcr, 0.10, 0.40)  # 0 = Bouba, 1 = Kiki
    gamma_spike = 0.12

    active_scalar = R0 * (1.0 + u_energy + chi * gamma_spike)
    active_r = active_scalar + (R0 * 0.1 * u_modes_field)

    # Idle heartbeat
    A_idle = 0.05
    idle_r = R0 * (0.85 + A_idle * np.sin(idle_freq * t))

    # Blend active vs idle
    final_r = G_active * active_r + (1.0 - G_active) * idle_r

    # PBR mapping
    tilt = float(attempt_feat.spectral_tilt[t_idx])
    hnr = float(attempt_feat.hnr_attempt[t_idx])
    roughness = _clamp(1.0 - hnr, 0.0, 1.0)
    metalness = _clamp(tilt * 1.5, 0.0, 1.0)
    spike_amt = _clamp(chi, 0.0, 1.0)

    base_color = _color_from_pitch(fp.mu_f0)
    colors = np.tile(base_color[None, :], (N, 1))

    return BubbleState(
        radii=final_r.astype(np.float32),
        colors=colors,
        pbr_props={
            "rough": roughness,
            "metal": metalness,
            "spike": spike_amt,
        },
    )


