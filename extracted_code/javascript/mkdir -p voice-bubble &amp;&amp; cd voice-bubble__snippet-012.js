const bgGeo = new THREE.SphereGeometry(7.5, 32, 24);
const bgMat = new THREE.MeshBasicMaterial({ color: 0x040611, side: THREE.BackSide });
const bg = new THREE.Mesh(bgGeo, bgMat);
