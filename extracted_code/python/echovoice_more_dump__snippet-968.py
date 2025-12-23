const res = await fetch(url);
const text = await res.text();
const vec = text.split(' ').map(w => w.length / 10);  # Simple embed
this.addNode([Math.random()*6, Math.random()*1, Math.random()*6], vec);  # New node from data
