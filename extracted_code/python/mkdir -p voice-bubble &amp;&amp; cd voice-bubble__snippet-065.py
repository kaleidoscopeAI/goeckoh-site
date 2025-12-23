// “Metalness” from spectral tilt (brighter/less negative tilt => more “metal”)
const metal = clamp(0.08 + (tilt + 2) / 4 * 0.65, 0.02, 0.9);
