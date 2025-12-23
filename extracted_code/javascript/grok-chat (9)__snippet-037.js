270      const pos = new THREE.Vector3(x * r, z * r, Math.sin(i) * r * 0.6);
271 -    const color = new THREE.Color(palette[i % palette.length]);
271 +    const color = new THREE.Color(colorForWord(w, palette[i % palette.l
     ength]));
272 +    const geom = geometryForWord(w);
273      targets.push({
    ⋮
278        energy: 0.6 + 0.04 * i,
278 -      metadata: { word: w }
279 +      metadata: { word: w, shape: geom }
280      });
    ⋮
296
297 +function colorForWord(word: string, fallback: string): string {
298 +  const map: Record<string, string> = {
299 +    dog: '#f97316',
300 +    car: '#0ea5e9',
301 +    cat: '#f59e0b',
302 +    tree: '#22c55e',
303 +    sky: '#38bdf8',
304 +    water: '#14b8a6',
305 +    fire: '#ef4444',
306 +    robot: '#94a3b8',
307 +  };
308 +  return map[word] || fallback;
309 +}
310 +
311 +function geometryForWord(word: string): 'sphere' | 'cube' | 'cone' | 't
     orus' | 'pyramid' {
312 +  const map: Record<string, 'sphere' | 'cube' | 'cone' | 'torus' | 'pyr
     amid'> = {
313 +    dog: 'sphere',
314 +    cat: 'sphere',
315 +    car: 'cube',
316 +    tree: 'cone',
317 +    fire: 'cone',
318 +    sky: 'torus',
319 +    robot: 'cube',
320 +    water: 'torus',
321 +  };
322 +  return map[word] || 'sphere';
323 +}
324 +
325  export default App;

