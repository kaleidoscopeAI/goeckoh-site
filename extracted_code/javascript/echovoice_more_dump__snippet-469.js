const vec = node.constructs[constructKeys[cIdx]] || new Array(5).fill(0);
const end = mesh.position.clone().add(new THREE.Vector3(vec[0], vec[1], vec[2]).multiplyScalar(CONSTRUCT_SCALE));
