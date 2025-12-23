if (!mount.current) return;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030711);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 5000);
camera.position.z = 800;

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
mount.current.appendChild(renderer.domElement);

const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(nodeCount * 3);
const colors = new Float32Array(nodeCount * 3);
const sizes = new Float32Array(nodeCount);

// Fibonacci sphere init
for (let i = 0; i < nodeCount; i++) {
  const i3 = i * 3;
  const phi = Math.acos(1 - 2 * (i + 0.5) / nodeCount);
  const theta = Math.PI * 2 * (i + 0.5) / (1 + Math.sqrt(5));
  positions[i3] = 400 * Math.sin(phi) * Math.cos(theta);
  positions[i3 + 1] = 400 * Math.sin(phi) * Math.sin(theta);
  positions[i3 + 2] = 400 * Math.cos(phi);
  colors[i3] = i % 2 === 0 ? 0.8 : 0.2;
  colors[i3 + 1] = 0.2;
  colors[i3 + 2] = i % 2 === 0 ? 0.2 : 0.8;
  sizes[i] = 2;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

const material = new THREE.ShaderMaterial({
  uniforms: {
    time: { value: 0 },
    emotionalValence: { value: 0 },
    emotionalArousal: { value: 0.5 },
    globalCoherence: { value: 0.5 }
  },
  vertexShader: `
    attribute float size;
    varying vec3 vColor;
    uniform float time;
    uniform float emotionalValence;
    uniform float emotionalArousal;
    void main() {
      vColor = color;
      float emotionalWave = sin(position.x * 0.01 + time * emotionalArousal) * emotionalValence;
      vec3 pos = position + emotionalWave * 10.0;
      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_PointSize = size * (300.0 / -mvPosition.z) * (1.0 + emotionalArousal);
      gl_Position = projectionMatrix * mvPosition;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    uniform float globalCoherence;
    void main() {
      float coherenceGlow = globalCoherence * 0.5;
      vec3 finalColor = vColor + vec3(coherenceGlow);
      vec2 coord = gl_PointCoord - vec2(0.5);
      if(length(coord) > 0.5) discard;
      gl_FragColor = vec4(finalColor, 1.0);
    }`,
  vertexColors: true,
  transparent: true
});

const points = new THREE.Points(geometry, material);
scene.add(points);

// Worker init
const worker = new Worker(new URL('../workers/particleWorker.ts', import.meta.url), { type: 'module' });
onWorkerReady(worker);

let lastBuffer: ArrayBuffer | null = null;
let positionAttribute = points.geometry.getAttribute('position') as THREE.BufferAttribute;
let frameId: number | null = null;
let lastFrameTime = performance.now();

worker.postMessage({ cmd: 'init', data: { count: nodeCount, positions: positions.buffer } }, [positions.buffer]);

worker.onmessage = (e) => {
  const { cmd } = e.data;
  if (cmd === 'positions') {
    const arr = new Float32Array(e.data.positions);
    // assign new buffer (zero-copy)
    positionAttribute.array = arr;
    positionAttribute.needsUpdate = true;
    // return previous buffer to worker
    if (lastBuffer) {
      worker.postMessage({ cmd: 'returnBuffer', buffer: lastBuffer }, [lastBuffer]);
    }
    lastBuffer = e.data.positions;
    // update UI
    if (e.data.systemState) onSystemUpdate(e.data.systemState);
    // continue render loop
    frameId = requestAnimationFrame(render);
  } else if (cmd === 'ready') {
    console.log('Worker ready');
  }
};

const updatePhysics = () => worker.postMessage({ cmd: 'update' });

const render = () => {
  const now = performance.now();
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  // adaptive density (scale node count based on render time)
  // simple example: if dt > 20ms -> request worker to lower active internal density (worker handles this param)
  renderer.render(scene, camera);
  updatePhysics();
};

const handleResize = () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
};
window.addEventListener('resize', handleResize);

// initial render start
frameId = requestAnimationFrame(render);

return () => {
  if (frameId) cancelAnimationFrame(frameId);
  worker.terminate();
  renderer.dispose();
  window.removeEventListener('resize', handleResize);
  // free DOM
  mount.current?.removeChild(renderer.domElement);
};
