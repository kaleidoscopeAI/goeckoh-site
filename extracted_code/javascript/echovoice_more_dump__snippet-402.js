const mat = new THREE.PointsMaterial({ size: 0.2, color: colorMap(Math.random()), transparent: true, opacity: 0.7 });
const points = new THREE.Points(geom, mat);
