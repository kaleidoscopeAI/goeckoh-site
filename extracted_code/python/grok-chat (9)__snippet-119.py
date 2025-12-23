 to SSE:

 const es = new EventSource('http://localhost:8080/mirror/stream');
 es.onmessage = (e) => {
   const { corrected, gcl } = JSON.parse(e.data);
   // drive your 3D scene from `corrected` text and modulate visuals with
