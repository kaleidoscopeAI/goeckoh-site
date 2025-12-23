<form id="fSeed" class="flex">
<input id="seedtxt" placeholder="Optional seed text to bias identity"/>
<button type="submit">Regenerate</button>
<button type="button" id="btnFreeze">Lock/Unlock</button>
</form>
<form id="fSpeak" class="flex">
<input id="say" placeholder="Make it speak this…"/>
<button type="submit">Speak</button>
<small>Writes onbrain/audio/identity_preview.wav</small>
</form>
</div>
</div>
</div>
<script>
async function loadIdent(){
const r=await fetch('/identity'); const j=await r.json();
document.getElementById('ident').textContent = JSON.stringify(j.identity,null,2);
document.getElementById('avatar').src='/identity/avatar.svg?'+Date.now();
}
document.getElementById('fSeed').onsubmit=async(e)=>{e.preventDefault();
const seed=document.getElementById('seedtxt').value;
await fetch('/identity/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed})});
document.getElementById('seedtxt').value='';
await loadIdent();
}
document.getElementById('btnFreeze').onclick=async()=>{
const cur=await (await fetch('/identity')).json();
const frozen = !!(cur.identity && cur.identity.locks && cur.identity.locks.frozen);
await fetch('/identity/lock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frozen:!frozen})});
await loadIdent();
}
document.getElementById('fSpeak').onsubmit=async(e)=>{e.preventDefault();
const text=document.getElementById('say').value;
if(!text.trim()) return;
await fetch('/identity/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
document.getElementById('say').value='';
}
loadIdent();
