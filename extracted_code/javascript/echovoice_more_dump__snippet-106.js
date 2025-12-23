export function retrieveClosest(input: number[]) {
  let minDist = Infinity, closest: number[] = [];
  for (const sub of faissIndex.values()) {
    const dist = len(sub.flat().map((v, i) => v - input[i % input.length]));
    if (dist < minDist) { minDist = dist; closest = sub.flat(); }
  }
  return closest;
