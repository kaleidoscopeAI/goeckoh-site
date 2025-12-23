const idx = Math.floor((i * count) / sampleSize) * 3;
const vy = velocities[idx];
const speed = Math.hypot(velocities[idx], velocities[idx + 1], velocities[idx + 2]);
totalVal += (vy > 0 ? 1 : -1) * speed;
totalArousal += speed;
