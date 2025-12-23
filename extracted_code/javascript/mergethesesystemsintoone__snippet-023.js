const { width, height } = ctx.canvas;
ctx.clearRect(0, 0, width, height);

// Set up 3D projection matrix
const perspective = 500;
const scale = 40;

// Project 3D point to 2D
const project = (point) => {
  const [x, y, z] = point;
  const rotX = math.cos(rotation.x) * x - math.sin(rotation.x) * z;
  const rotZ = math.sin(rotation.x) * x + math.cos(rotation.x) * z;
  const rotY = math.cos(rotation.y) * y - math.sin(rotation.y) * rotZ;
  const projZ = math.sin(rotation.y) * y + math.cos(rotation.y) * rotZ;

  const projectScale = scale / (perspective - projZ);
  return [
    width/2 + rotX * projectScale,
    height/2 + rotY * projectScale,
    projZ
  ];
};

// Draw tension field
if (state.tension_field) {
  const { X, Y, Z, tension } = state.tension_field;
  const maxTension = math.max(tension.flat());

  // Sample points for visualization
  const sampleRate = 5;
  for (let i = 0; i < X.length; i += sampleRate) {
    for (let j = 0; j < Y.length; j += sampleRate) {
      for (let k = 0; k < Z.length; k += sampleRate) {
        const point = [X[i], Y[j], Z[k]];
        const t = tension[i][j][k] / maxTension;
        const [px, py] = project(point);

        ctx.fillStyle = `rgba(255, 0, 0, ${t * 0.2})`;
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

// Draw connections
ctx.strokeStyle = 'rgba(100, 100, 255, 0.5)';
ctx.lineWidth = 1;

for (const [i, j] of state.connections) {
  const point1 = state.points[i];
  const point2 = state.points[j];

  const [x1, y1] = project(point1[0]);
  const [x2, y2] = project(point2[0]);

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

// Draw memory points
for (const [position, energy, activation] of state.points) {
  const [px, py, pz] = project(position);
  const radius = 5 + energy * 2;

  // Point glow based on activation
  const gradient = ctx.createRadialGradient(px, py, 0, px, py, radius * 2);
  gradient.addColorStop(0, `rgba(0, 255, 255, ${activation})`);
  gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(px, py, radius, 0, 2 * Math.PI);
  ctx.fill();

  // Point core
  ctx.fillStyle = `rgb(0, ${Math.floor(energy * 255)}, 255)`;
  ctx.beginPath();
  ctx.arc(px, py, radius/2, 0, 2 * Math.PI);
  ctx.fill();
}

const canvas = canvasRef.current;
const ctx = canvas.getContext('2d');

// Handle rotation with mouse
const handleMouseMove = (e) => {
  if (e.buttons === 1) {
    setRotation(prev => ({
      x: prev.x + e.movementX * 0.01,
      y: prev.y + e.movementY * 0.01
    }));
  }
};

canvas.addEventListener('mousemove', handleMouseMove);

// Animation loop
let animationFrame;
const animate = () => {
  if (cubeState) {
    drawCube(ctx, cubeState);
  }
  animationFrame = requestAnimationFrame(animate);
};
animate();

return () => {
  canvas.removeEventListener('mousemove', handleMouseMove);
  cancelAnimationFrame(animationFrame);
};

const updateInterval = setInterval(() => {
  // Simulate cube state changes
  const points = Array(10).fill(0).map(() => [
    [Math.random() * 10, Math.random() * 10, Math.random() * 10],
    Math.random(),
    Math.random()
  ]);

  const connections = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      if (Math.random() < 0.3) {
        connections.push([i, j]);
      }
    }
  }

  setCubeState({
    points,
    connections,
    tension_field: {
      X: Array(10).fill(0),
      Y: Array(10).fill(0),
      Z: Array(10).fill(0),
      tension: Array(10).fill(Array(10).fill(Array(10).fill(Math.random())))
    }
  });
}, 50);

return () => clearInterval(updateInterval);

<div className="p-4">
  <h2 className="text-xl font-bold mb-4">Cube Memory System Visualization</h2>
  <div className="relative">
    <canvas
      ref={canvasRef}
      width={800}
      height={600}
      className="border border-gray-300 rounded"
    />
    <div className="absolute top-2 left-2 text-sm text-gray-600">
      Drag to rotate
    </div>
  </div>
</div>

