  import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js";
  import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js";
  const container = document.getElementById('canvas'); const state = document.getElementById('state'); const logEl = 
document.getElementById('log');
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
renderer.setSize(container.clientWidth, container.clientHeight);} window.addEventListener('resize', resize);

  function log(x){const p=document.createElement('div'); p.textContent=x; logEl.appendChild(p); logEl.scrollTop=logEl.scrollHeight;}

  // Text WS for events
  function wsText(){
    const proto = (location.protocol==='https:')?'wss':'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (ev)=>{ try{const msg=JSON.parse(ev.data); if(msg.type==='caption'){log('Caption: '+(msg.data.caption||''));} }catch{} };
    ws.onopen = ()=>log('ws:/ws connected'); ws.onclose=()=>log('ws:/ws closed');
  } wsText();

  // Binary WS for avatar
  function wsAvatar(){
    const proto = (location.protocol==='https:')?'wss':'ws';
    const ws = new WebSocket(`${proto}://${location.host}/avatar`);
    ws.binaryType='arraybuffer';
    let last=0;
    ws.onopen=()=>{state.textContent='connected';};
    ws.onclose=()=>{state.textContent='disconnected – retrying'; setTimeout(wsAvatar, 1500);};
    ws.onmessage=(ev)=>{
      const buf=ev.data; if(!(buf instanceof ArrayBuffer)) return;
      const u32=new Uint32Array(buf,0,1); const n=u32[0]>>>0; const f32=new Float32Array(buf,4);
      if(f32.length<n*3) return; ensureCapacity(n); positions.set(f32.subarray(0,n*3)); geom.attributes.position.needsUpdate=true;
      if(n!==last){state.textContent=`connected • ${n.toLocaleString()} points`; last=n;} updateColors();
    };
  } wsAvatar();

  document.getElementById('speak').onclick=async ()=>{
    const r=await fetch('/recent?table=captions&limit=1'); const js=await r.json();
    if(js && js.rows && js.rows[0]){const text=js.rows[0][3]||''; log('Speak: '+text);}
  };
  document.getElementById('seed').onclick=async ()=>{await fetch('/ingest', {method:'POST'}); log('Local seed added');};
