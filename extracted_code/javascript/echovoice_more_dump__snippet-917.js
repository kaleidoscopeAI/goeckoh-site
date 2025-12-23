const mount = mountRef.current;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(75, mount.clientWidth / mount.clientHeight, 0.1, 2000);
camera.position.z = 600;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(mount.clientWidth, mount.clientHeight);
mount.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const geometry = new THREE.SphereGeometry(1.5, 8, 8);
const material = new THREE.MeshBasicMaterial({ color: 0x00ffff });
const group = new THREE.Group();
scene.add(group);

const numNodes = 8000;
const nodes = [];

for (let i = 0; i < numNodes; i++) {
  const node = new THREE.Mesh(geometry, material.clone());
  node.position.set(
    (Math.random() - 0.5) * 1000,
    (Math.random() - 0.5) * 1000,
    (Math.random() - 0.5) * 1000
  );
  node.material.color.setHSL(Math.random(), 1.0, 0.5);
  nodes.push(node);
  group.add(node);
}

function createVisualEmbedding(thought) {
  const hash = Array.from(thought).reduce((acc, c) => acc + c.charCodeAt(0), 0);
  const vectors = nodes.map((_, i) => {
    const t = (i / numNodes) * Math.PI * 8;
    const r = 300 + 100 * Math.sin(hash * 0.01 + i * 0.02);
    const twist = Math.sin(hash * 0.005) * 2.0;
    return new THREE.Vector3(
      r * Math.cos(t + twist),
      r * Math.sin(t + twist),
      200 * Math.sin(i * 0.01 + hash * 0.03)
    );
  });
  return vectors;
}

let targetPositions = createVisualEmbedding(aiThought || 'initial');

let time = 0;
function animate() {
  time += 0.01;
  nodes.forEach((node, i) => {
    const target = targetPositions[i];
    node.position.lerp(target, 0.02 * metrics.coherence);
    const hue = (metrics.valence * 0.5 + metrics.curiosity * 0.5 + Math.sin(i * 0.1 + time)) % 1.0;
    node.material.color.setHSL(hue, 1.0, 0.5);
  });

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

const handleResize = () => {
  camera.aspect = mount.clientWidth / mount.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(mount.clientWidth, mount.clientHeight);
};

window.addEventListener('resize', handleResize);

const interval = setInterval(() => {
  if (aiThought) {
    targetPositions = createVisualEmbedding(aiThought);
  }
}, 4000);

return () => {
  mount.removeChild(renderer.domElement);
  window.removeEventListener('resize', handleResize);
  clearInterval(interval);
};
