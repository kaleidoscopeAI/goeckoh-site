document.getElementById('pSpeak').onclick=async()=>{
const caps = await (await fetch('/recent?table=captions&limit=1')).json();
const text = caps.rows && caps.rows[0] ? (caps.rows[0][3]||"I am crystallizing what I learn.") : "I am crystallizing what I learn.";
await fetch('/persona/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
document.getElementById('pAvatar').src='/persona/avatar.svg?'+Date.now();
};
document.getElementById('pNudge').onclick=async()=>{
// taking over happens automatically each anneal; this just forces a status refresh
const _=await fetch('/persona/layout');
document.getElementById('pAvatar').src='/persona/avatar.svg?'+Date.now();
};
