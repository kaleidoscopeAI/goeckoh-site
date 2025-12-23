const nodes: THREE.Mesh[] = [];
const constructs: THREE.Line[][] = [];
const nodeParticles: THREE.Points[][] = [];
const nodeGeometry = new THREE.SphereGeometry(1, 32, 32);
const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x8888ff, transparent: true, opacity: 0.8 });
for (let i = 0; i < NODE_COUNT; i++) {
const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
