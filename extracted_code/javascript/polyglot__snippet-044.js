async function poll(){
const s=await (await fetch('/status')).json();
document.getElementById('st').textContent=JSON.stringify(s,null,2);
if(s.latest_frame){document.getElementById('im').src='/frame?path='+encodeURIComponent(s.latest_frame)+'&t='+Date.now()}
setTimeout(poll, 1200);
}
document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('q').value; if(!text.trim())return;
const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
poll()
