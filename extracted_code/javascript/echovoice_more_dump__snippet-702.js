const tex = makeLabelTexture(c.name, 128, 32, "#fff", "#222");
const map = new THREE.CanvasTexture(tex);
const mat = new THREE.SpriteMaterial({ map, transparent: true });
const sprite = new THREE.Sprite(mat);
