async function refresh(){
const r=await fetch('/status'); const j=await r.json(); document.getElementById('status').textContent=JSON.stringify(j,null,2)}
document.getElementById('fTeach').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('teach').value; if(!text.trim())return;
const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('teachOut').innerHTML='Taught fact id: '+j.id; document.getElementById('teach').value='';
refresh()}
document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('q').value; if(!text.trim())return;
const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
document.getElementById('btnAuto').onclick=async()=>{
const s=await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({autonomous:'toggle'})}); await
refresh()}
document.getElementById('btnSeed').onclick=async()=>{await fetch('/seed',{method:'POST'}); await refresh()}
refresh()
