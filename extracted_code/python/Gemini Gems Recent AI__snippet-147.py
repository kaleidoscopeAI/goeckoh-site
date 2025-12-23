// Fetch molecules from the API
const fetchMolecules = async () => {
  try {
    const response = await axios.get('/api/molecules');
    setMolecules(response.data);
  } catch (error) {
    console.error("Error fetching molecules:", error);
  }
};

fetchMolecules();

// Three.js setup (simplified)
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
mountRef.current.appendChild(renderer.domElement);

// Add lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(1, 1, 1).normalize();
scene.add(directionalLight);

// Create 3D objects (spheres - simplified)
molecules.forEach(molecule => {
    const geometry = new THREE.SphereGeometry(0.1, 32, 32);
    const material = new THREE.MeshStandardMaterial({ color: 0xffa500 }); // Use MeshStandardMaterial for lighting
    const sphere = new THREE.Mesh(geometry, material);

    // Placeholder positions (replace with data from API)
    sphere.position.x = Math.random() * 10 - 5;
    sphere.position.y = Math.random() * 10 - 5;
    sphere.position.z = Math.random() * 10 - 5;

    scene.add(sphere);
});

camera.position.z = 5;

const animate = function () {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
};

animate();

return () => {
    mountRef.current.removeChild(renderer.domElement);
};
