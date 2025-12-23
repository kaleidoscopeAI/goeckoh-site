  <div class="card"><h3>Recent</h3><a href="/recent?table=facts" target="_blank">facts</a> · <a href="/recent?table=energetics"
  target="_blank">energetics</a> · <a href="/recent?table=captions" target="_blank">captions</a></div>
  <script>
  document.getElementById('fTeach').onsubmit=async(e)=>{e.preventDefault()
   const text=document.getElementById('teach').value; if(!text.trim())return;
   const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
   const j=await r.json(); document.getElementById('teachOut').innerHTML='Taught fact id: '+j.id; document.getElementById('teach').value=''}
  document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
   const text=document.getElementById('q').value; if(!text.trim())return;
   const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
   const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
