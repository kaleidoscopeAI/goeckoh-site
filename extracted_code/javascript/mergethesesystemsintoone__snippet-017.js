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
