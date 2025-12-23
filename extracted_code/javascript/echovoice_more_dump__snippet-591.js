const tex = makeLabelTexture(c.name + ` (${c.activation.toFixed(2)})`, 128, 32, "#fff", activationColor(c.activation));
const map = new THREE.CanvasTexture(tex);
