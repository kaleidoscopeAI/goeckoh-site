const curve = new THREE.CatmullRomCurve3(pts);
const points = curve.getPoints(40);
const geom = new THREE.BufferGeometry().setFromPoints(points);
const intensity = Math.min(1, Math.abs(a.activation) + Math.abs(b.activation));
const color = new THREE.Color().setHSL(0.08 * intensity, 0.9, 0.5);
const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 + 0.6 * intensity });
const line = new THREE.Line(geom, mat);
