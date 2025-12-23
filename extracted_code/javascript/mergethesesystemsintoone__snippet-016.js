const proto = (location.protocol==='https:')?'wss':'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);
ws.onmessage = (ev)=>{ try{const msg=JSON.parse(ev.data); if(msg.type==='caption'){log('Caption: '+(msg.data.caption||''));} }catch{} };
ws.onopen = ()=>log('ws:/ws connected'); ws.onclose=()=>log('ws:/ws closed');
