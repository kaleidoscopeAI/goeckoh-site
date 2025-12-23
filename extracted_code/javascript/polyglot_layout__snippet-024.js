 const s=await (await fetch('/status')).json();
 document.getElementById('st').textContent=JSON.stringify(s,null,2);
 if(s.latest_frame){document.getElementById('im').src='/frame?path='+encodeURIComponent(s.latest_frame)+'&t='+Date.now()}
 setTimeout(poll, 1200);
