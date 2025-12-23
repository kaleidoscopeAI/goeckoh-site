const dist = len(sub.flat().map((v, i) => v - input[i % input.length]));
if (dist < minDist) { minDist = dist; closest = sub.flat(); }
