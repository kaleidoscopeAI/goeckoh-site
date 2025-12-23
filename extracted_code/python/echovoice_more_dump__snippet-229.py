import p5 from 'p5';

const App = () => {
  const sketch = (p: p5) => {
    let particles: any[] = [];  // From websocket snapshot

    p.setup = () => {
      p.createCanvas(800, 600, p.WEBGL);  // 3D
      // WS connect to server:4000 for snapshots
      const ws = new WebSocket('ws://localhost:4000');
      ws.onmessage = (msg) => particles = JSON.parse(msg.data).particles;
    };

    p.draw = () => {
      p.background(0);  // Starry space
      p.orbitControl();  // 3D cam
      particles.forEach(part => {
        p.push();
        p.translate(...part.pos);
        p.fill(part.color);
        p.sphere(part.size);  // Node sphere
        if (part.halo) {  // Low E halo
          p.noFill();
          p.stroke(part.color.replace('50%', '30%'));
          p.sphere(part.size * 1.5);  // Halo ring
        }
        p.pop();
      });
      // Dynamic evolution: mutate pos based on bonds (N-body client-side approx)
    };
  };

  return <p5.Sketch sketch={sketch} />;
