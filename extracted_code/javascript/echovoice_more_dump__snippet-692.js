const constructs = svc.getConstructs();
const baseScale = 1.5;
for (const c of constructs) {
const tex = makeLabelTexture(c.name, 128, 32, "#fff", "#222");
const map = new THREE.CanvasTexture(tex);
const mat = new THREE.SpriteMaterial({ map, depthTest: false, depthWrite: false, transparent: true });
const sprite = new THREE.Sprite(mat);
