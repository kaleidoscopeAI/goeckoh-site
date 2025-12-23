const socket = io('/kaleido');

let scene, camera, renderer, points, positionAttr;
const NODE_COUNT = 20000;

init();
function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 1, 5000);
  camera.position.z = 800;
  renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(NODE_COUNT * 3);
  const colors = new Float32Array(NODE_COUNT * 3);
  for (let i=0;i<NODE_COUNT;i++){
    positions[i*3] = 0; positions[i*3+1]=0; positions[i*3+2]=0;
    colors[i*3]=0.5; colors[i*3+1]=0.2; colors[i*3+2]=0.8;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions,3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors,3));
  const material = new THREE.PointsMaterial({ size:2, vertexColors:true });
  points = new THREE.Points(geometry, material);
  scene.add(points);
  positionAttr = points.geometry.getAttribute('position');

  animate();
}

socket.on('positions', (data) => {
  // positions come as bytes; reconstruct Float32Array
  const buf = new Uint8Array(data.positions);
  const floatBuf = new Float32Array(buf.buffer);
  positionAttr.array = floatBuf;
  positionAttr.needsUpdate = true;
});

function animate(){
  requestAnimationFrame(animate);
  scene.rotation.y += 0.001;
  renderer.render(scene, camera);
}
