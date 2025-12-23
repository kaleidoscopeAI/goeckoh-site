const [x,y] = worldToScreen(c.coord.x, c.coord.y);
const radius = Math.max(6, 8 + Math.log(1 + Math.abs(c.activation)) * 6);
const color = colorForActivation(c.activation);
