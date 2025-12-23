const nodeVectors: THREE.ArrowHelper[] = [];
for (let j = 0; j < 5; j++) {
const dir = new THREE.Vector3(Math.random(), Math.random(), Math.random()).normalize();
const origin = nodeMesh.position.clone();
const arrow = new THREE.ArrowHelper(dir, origin, 0, 0xff0000, 0.3, 0.2);
