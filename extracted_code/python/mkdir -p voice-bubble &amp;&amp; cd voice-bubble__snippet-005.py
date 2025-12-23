// Material: roughness from inverse HNR (more noise => rougher)
const rough = clamp(0.08 + (1.0 - clamp((hnr + 5) / 30, 0, 1)) * 0.75, 0.04, 0.95);
