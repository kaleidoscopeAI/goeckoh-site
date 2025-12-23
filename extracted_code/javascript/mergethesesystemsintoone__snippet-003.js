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
