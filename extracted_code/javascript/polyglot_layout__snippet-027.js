const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0c10);
const camera = new THREE.PerspectiveCamera(60, container.clientWidth/container.clientHeight, 0.01, 100);
camera.position.set(0,0,3.2);
const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight); container.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
let points=null, positions=null, colors=null; const geom=new THREE.BufferGeometry();
const material = new THREE.PointsMaterial({ size:0.01, vertexColors:true, opacity:0.95, transparent:true });
function ensureCapacity(n){const n3=n*3; if(!positions || positions.length!==n3){positions=new Float32Array(n3); colors=new Float32Array(n3);
  geom.setAttribute('position', new THREE.BufferAttribute(positions,3).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute('color', new THREE.BufferAttribute(colors,3).setUsage(THREE.DynamicDrawUsage));
  if(!points){points=new THREE.Points(geom, material); scene.add(points);}}}
function updateColors(){for(let i=0;i<positions.length;i+=3){const x=positions[i], y=positions[i+1], z=positions[i+2];
  const r=Math.min(1.0, Math.sqrt(x*x+y*y+z*z)); const t=r; const R=0.2+0.8*t; const G=0.5+0.5*(1.0-Math.abs(t-0.5)*2.0); const B=1.0-0.8*t;
  colors[i]=R; colors[i+1]=G; colors[i+2]=B;} geom.attributes.color.needsUpdate=true;}
function render(){controls.update(); renderer.render(scene, camera); requestAnimationFrame(render);} render();
function resize(){camera.aspect=container.clientWidth/container.clientHeight; camera.updateProjectionMatrix();
