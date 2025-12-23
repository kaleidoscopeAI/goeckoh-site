// Add light
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(1, 1, 1).normalize();
scene.add(light);

// Create bonds geometry and material
const bondsGeometry = new THREE.BufferGeometry();
const bondsMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 1 });
const bondsLines = new THREE.LineSegments(bondsGeometry, bondsMaterial);
scene.add(bondsLines);

// Store the bonds geometry for later update
viz.bondsGeometry = bondsGeometry;

return { scene, camera, renderer, nodesMesh, bondsGeometry };

