const r=await fetch('/recent?table=captions&limit=1'); const js=await r.json();
if(js && js.rows && js.rows[0]){const text=js.rows[0][3]||''; log('Speak: '+text);}
