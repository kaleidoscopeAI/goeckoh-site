    const n3 = npoints * 3;
    if (!positions || positions.length !== n3) {
      positions = new Float32Array(n3);
      colors = new Float32Array(n3);
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
      if (!points) {
        points = new THREE.Points(geom, material);
        scene.add(points);
      }
    }
}

function updateColors() {
  // Map radius to color (center = cool blue, edge = warm)
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i+1], z = positions[i+2];
    const r = Math.min(1.0, Math.sqrt(x*x+y*y+z*z));
    // simple palette
    const t = r;
    const R = 0.2 + 0.8 * t;
    const G = 0.5 + 0.5 * (1.0 - Math.abs(t-0.5)*2.0);
    const B = 1.0 - 0.8 * t;
    colors[i] = R; colors[i+1] = G; colors[i+2] = B;
  }
  geom.attributes.color.needsUpdate = true;
}

function render() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
render();

function wsURL() {
  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  const port = location.port || (location.protocol === 'https:' ? '443' : '80');
  // Default to same host/port; if served from file:// use localhost:8767
  let host = location.hostname;
  let p = location.port;
  if (location.protocol === 'file:') { host = 'localhost'; p = '8767'; }
  return `${proto}://${host}:${p || port}/avatar`;
}

function connect() {
  const url = wsURL();
  wsurlEl.textContent = url;
  const ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  statusEl.textContent = 'connecting…';

    let lastCount = 0;

    ws.onopen = () => { statusEl.textContent = 'connected'; };
    ws.onclose = () => { statusEl.textContent = 'disconnected – retrying in 2s'; setTimeout(connect, 2000); };
    ws.onerror = (e) => { statusEl.textContent = 'error – see console'; console.error(e); };

    ws.onmessage = (ev) => {
       if (typeof ev.data === 'string') {
         // JSON fallback
         try {
           const msg = JSON.parse(ev.data);
           if (msg && msg.type === 'avatar' && Array.isArray(msg.positions)) {
             const arr = msg.positions;
             const n = Math.floor(arr.length / 3);
             ensureCapacity(n);
             for (let i=0;i<n*3;i++) positions[i] = arr[i];
             geom.attributes.position.needsUpdate = true;
             if (n !== lastCount) { statusEl.textContent = `connected • ${n.toLocaleString()} points (JSON)`; lastCount = n; }
             updateColors();
           }
         } catch {}
         return;
       }
       // Binary fast path: [uint32 N] + [N*3 float32]
       const buf = ev.data;
       if (!(buf instanceof ArrayBuffer)) return;
       const u32 = new Uint32Array(buf, 0, 1);
       const n = u32[0] >>> 0;
       const f32 = new Float32Array(buf, 4);
       if (f32.length < n*3) return;
       ensureCapacity(n);
       positions.set(f32.subarray(0, n*3));
       geom.attributes.position.needsUpdate = true;
       if (n !== lastCount) { statusEl.textContent = `connected • ${n.toLocaleString()} points`; lastCount = n; }
       updateColors();
    };
}

connect();

// Resize
function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', onResize);

// Reset
document.getElementById('reset').addEventListener('click', (e) => {
  e.preventDefault();
  camera.position.set(0,0,3.2);
  controls.reset();
});
