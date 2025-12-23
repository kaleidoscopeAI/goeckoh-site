// Load bubble mesh
const geometry = new THREE.IcosahedronGeometry(1.0, 3);

// Per-frame update
function animate(frameIdx) {
    const state = bubbleStates[frameIdx];
    
    // Update vertex positions
    for (let i = 0; i < vertices.length; i++) {
        const pos = baseVertices[i].clone();
        const normal = normals[i];
        pos.add(normal.multiplyScalar(state.radii[i]));
        geometry.attributes.position.array[i*3] = pos.x;
        geometry.attributes.position.array[i*3+1] = pos.y;
        geometry.attributes.position.array[i*3+2] = pos.z;
    }
    geometry.attributes.position.needsUpdate = true;
    
    // Update material uniforms
    material.uniforms.uRoughness.value = state.pbr_props.rough;
    material.uniforms.uMetalness.value = state.pbr_props.metal;
    material.uniforms.uSpikeAmount.value = state.pbr_props.spike;
}
