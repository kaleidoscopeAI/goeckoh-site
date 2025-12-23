const idx = Math.floor((i * count) / sampleSize);
const v = idx * 3;
const speed = Math.sqrt(velocities[v]**2 + velocities[v+1]**2 + velocities[v+2]**2);
totalValence += (velocities[v] > 0 ? 1 : -1) * speed;
totalArousal += speed;
