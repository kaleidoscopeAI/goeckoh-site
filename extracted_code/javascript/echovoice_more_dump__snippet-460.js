const nodeGeometry = new THREE.SphereGeometry(1, 32, 32);
const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x8888ff, transparent: true, opacity: 0.8 });
const nodes: THREE.Mesh[] = [];
const vectors: THREE.ArrowHelper[][] = [];
const constructs: THREE.Line[][] = [];
for (let i = 0; i < NODE_COUNT; i++) {
const nodeMesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
