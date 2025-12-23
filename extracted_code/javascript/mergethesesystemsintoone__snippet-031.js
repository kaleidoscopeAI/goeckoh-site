<div class="card"><h3>Persona</h3>
<div style="display:flex;gap:12px;align-items:center">
<img id="avatar" src="/persona/avatar.svg" width="128" height="128" style="border-radius:16px;border:1px solid #eee"/>
<div style="flex:1">
<button id="speak">Speak last caption</button>
<a href="/persona/layout" target="_blank">layout</a> · <a href="/metrics/coherence" target="_blank">coherence</a>
</div>
</div>
</div>
<div class="card"><h3>Recent</h3><a href="/recent?table=facts" target="_blank">facts</a> · <a href="/recent?table=energetics"
target="_blank">energetics</a> · <a href="/recent?table=captions" target="_blank">captions</a> · <a href="/metrics/coherence"
target="_blank">coherence</a></div>
<script>
document.getElementById('fTeach').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('teach').value; if(!text.trim())return;
const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('teachOut').innerHTML='Taught fact id: '+j.id; document.getElementById('teach').value=''}
document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
const text=document.getElementById('q').value; if(!text.trim())return;
const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
document.getElementById('speak').onclick=async()=>{
const caps = await (await fetch('/recent?table=captions&limit=1')).json();
const text = caps.rows && caps.rows[0] ? (caps.rows[0][3]||"I am crystallizing what I learn.") : "I am crystallizing what I learn.";
await fetch('/persona/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
document.getElementById('avatar').src='/persona/avatar.svg?'+Date.now();
}
