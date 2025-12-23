const nodeConstructs: THREE.Line[] = [];
for (let k = 0; k < 9; k++) {
const points = [mesh.position.clone(), mesh.position.clone()];
const geometry = new THREE.BufferGeometry().setFromPoints(points);
const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xffff00, transparent: true, opacity: 0.5 }));
