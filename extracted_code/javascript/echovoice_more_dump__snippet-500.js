const speciesParticles: THREE.Points[] = [];
for (let s = 0; s < 5; s++) {
const positions = new Float32Array(PARTICLE_COUNT * 3);
for (let p = 0; p < PARTICLE_COUNT; p++) {
