async function refresh(){
const r=await fetch('/status'); const j=await r.json(); document.getElementById('status').textContent=JSON.stringify(j,null,2)}
document.getElementById('fTeach').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('teach').value; if(!text.trim())return;
const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('teachOut').innerHTML='Taught id: '+j.id; document.getElementById('teach').value=''; refresh()}
document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('q').value; if(!text.trim())return;
const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
document.getElementById('btnAuto').onclick=async()=>{await fetch('/config',{method:'POST',headers:{'Content-
Type':'application/json'},body:JSON.stringify({autonomous:'toggle'})}); await refresh()}
document.getElementById('btnNet').onclick=async()=>{await fetch('/net/config',{method:'POST',headers:{'Content-
Type':'application/json'},body:JSON.stringify({net:'toggle'})}); await refresh()}
document.getElementById('fSeed').onsubmit=async(e)=>{e.preventDefault()
const u=document.getElementById('seed').value.trim(); if(!u)return;
await fetch('/net/seed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({urls:[u]})});
document.getElementById('seed').value=''}
document.getElementById('btnConfig').onclick=async()=>{const a=document.getElementById('allow').value
await fetch('/net/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({allowlist:a})}); await refresh()}
refresh()
