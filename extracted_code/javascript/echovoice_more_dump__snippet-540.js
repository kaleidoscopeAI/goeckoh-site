const bounds = useMemo(() => {
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
for (const c of constructs) {
