307    };
308 -  return map[word] || fallback;
308 +  if (map[word]) return map[word];
309 +  // Hash any word into a gentle color
310 +  let h = 0;
311 +  for (let i = 0; i < word.length; i++) h = (h * 31 + word.charCodeAt(i
     )) >>> 0;
312 +  const hue = h % 360;
313 +  const sat = 55 + (h % 20); // 55–75
314 +  const light = 50 + (h % 10); // 50–59
315 +  return `hsl(${hue}, ${sat}%, ${light}%)` || fallback;
316  }
    ⋮
328    };
322 -  return map[word] || 'sphere';
329 +  if (map[word]) return map[word];
330 +  const shapes: Array<'sphere' | 'cube' | 'cone' | 'torus' | 'pyramid'>
      = ['sphere', 'cube', 'cone', 'torus', 'pyramid'];
331 +  let h = 0;
332 +  for (let i = 0; i < word.length; i++) h = (h * 33 + word.charCodeAt(i
     )) >>> 0;
333 +  return shapes[h % shapes.length];
334  }

