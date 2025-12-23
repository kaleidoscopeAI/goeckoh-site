const text=document.getElementById('q').value; if(!text.trim())return;
const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
