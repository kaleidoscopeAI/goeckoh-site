const aFast = 0.25;   // visual response
const aSlow = 0.12;   // for pitch \& tilt

const e = clamp(newFeat.energy ?? feat.energy, 0, 1);
const z = clamp(newFeat.zcr ?? feat.zcr, 0, 1);
const t = clamp(newFeat.tilt ?? feat.tilt, -2, 2);

