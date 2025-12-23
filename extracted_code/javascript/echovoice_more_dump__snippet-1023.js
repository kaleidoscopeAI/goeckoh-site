// Update bonds
const bonds = crystal.core.bonds; // Assume we have this
const positions = [];
for (let bond of bonds) {
    const node1 = nodes[bond[0]];
    const node2 = nodes[bond[1]];
    positions.push(node1.x, node1.y, node1.z);
    positions.push(node2.x, node2.y, node2.z);
}
viz.bondsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

