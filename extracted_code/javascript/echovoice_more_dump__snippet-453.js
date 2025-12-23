const angle = angles[i];
const val = (draft[name] ?? 0);
const handleRadius = radius * (0.25 + 0.65 * ((val + 1) / 2)); // map -1..1 -> 0.25..0.9 radially
const [hx, hy] = polarToXY(cx, cy, handleRadius, angle);
const lineEnd = polarToXY(cx, cy, radius, angle);
