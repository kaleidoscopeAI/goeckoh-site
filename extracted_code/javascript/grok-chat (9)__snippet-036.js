248
249 +function makeTargetsFromText(text: string): NodeTarget[] {
250 +  const palette = [
251 +    '#22c55e', '#38bdf8', '#f472b6', '#f97316', '#c084fc',
252 +    '#eab308', '#14b8a6', '#94a3b8', '#8b5cf6', '#ef4444'
253 +  ];
254 +  const words = text
255 +    .toLowerCase()
256 +    .split(/[^a-z0-9]+/)
257 +    .filter(Boolean)
258 +    .slice(0, 12);
259 +
260 +  const targets: NodeTarget[] = [];
261 +  const golden = Math.PI * (3 - Math.sqrt(5));
262 +  const r = 4;
263 +
264 +  words.forEach((w, i) => {
265 +    const theta = i * golden;
266 +    const z = 1 - (i / Math.max(1, words.length - 1)) * 2;
267 +    const radius = Math.sqrt(1 - z * z);
268 +    const x = Math.cos(theta) * radius;
269 +    const y = Math.sin(theta) * radius;
270 +    const pos = new THREE.Vector3(x * r, z * r, Math.sin(i) * r * 0.6);
271 +    const color = new THREE.Color(palette[i % palette.length]);
272 +    targets.push({
273 +      position: pos,
274 +      color,
275 +      type: 'thought',
276 +      awareness: 0.6 + 0.03 * i,
277 +      energy: 0.6 + 0.04 * i,
278 +      metadata: { word: w }
279 +    });
280 +  });
281 +
282 +  // If no words, keep a calm center node
283 +  if (!targets.length) {
284 +    targets.push({
285 +      position: new THREE.Vector3(0, 0, 0),
286 +      color: new THREE.Color('#94a3b8'),
287 +      type: 'thought',
288 +      awareness: 0.5,
289 +      energy: 0.5,
290 +      metadata: { word: 'silence' }
291 +    });
292 +  }
293 +  return targets;
294 +}
295 +
296  export default App;

