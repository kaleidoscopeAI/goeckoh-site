import p5 from 'p5';

const App = () => {
  const sketch = (p: p5) => {
    let particles: any[] = [];
    let stars: {x: number, y: number, z: number}[] = [];  // Starfield

    p.setup = () => {
      p.createCanvas(800, 600, p.WEBGL);
      // Starfield: 1000 stars, inspired dwmkerr/starfield
      for (let i = 0; i < 1000; i++) {
        stars.push({ x: p.random(-p.width, p.width), y: p.random(-p.height, p.height), z: p.random(100, 1000) });
      }
      // WS for snapshots
      const ws = new WebSocket('ws://localhost:4000');
      ws.onmessage = (msg) => particles = JSON.parse(msg.data).particles;
    };

    p.draw = () => {
      p.background(0);
      p.orbitControl();

      // Aurora gradient background: lerpColor wave, from tutorial
      for (let y = 0; y < p.height; y++) {
        const inter = p.map(y, 0, p.height, 0, 1) + p.sin(p.frameCount * 0.01 + y * 0.01) * 0.2;  // Wave
        const c = p.lerpColor(p.color(0, 50, 100), p.color(120, 50, 100), inter);
        p.stroke(c);
        p.line(-p.width/2, y - p.height/2, p.width/2, y - p.height/2);  // Horizontal lines
      }

      // Starfield: parallax, size~1/z
      p.noStroke();
      stars.forEach(s => {
        const sx = p.map(s.x / s.z, -1, 1, -p.width/2, p.width/2);
        const sy = p.map(s.y / s.z, -1, 1, -p.height/2, p.height/2);
        const size = p.map(s.z, 100, 1000, 2, 0.5);
        p.fill(255, p.map(s.z, 100, 1000, 255, 100));  // Dimmer farther
        p.circle(sx, sy, size);
        s.z -= 1;  // Move toward
        if (s.z < 100) s.z = 1000;  // Reset
      });

      particles.forEach((part, i) => {
        p.push();
        p.translate(...part.pos);
        p.fill(part.color);

        // N-body approx client: gravity to others, from Adesh_Brave sketch
        let acc = [0,0,0];
        particles.forEach((other, j) => {
          if (i === j) return;
          const r = p5.Vector.sub(p.createVector(...other.pos), p.createVector(...part.pos));
          const dist = r.mag() + 1e-9;
          acc = p5.Vector.add(acc, r.normalize().mult(-1 / dist**2));
        });
        // Update vel/pos (approx, full in server)
        part.vel = part.vel || [0,0,0];
        part.vel = p5.Vector.add(part.vel, acc.mult(0.01));
        part.pos = p5.Vector.add(part.pos, part.vel.mult(0.01));

        p.sphere(part.size);

        // Halo: stroke sphere, from shininess ref
        if (part.halo) {
          p.noFill();
          p.stroke(part.color.replace('50%', '30%'));
          p.shininess(50);  // Shiny glow
          p.sphere(part.size * 1.5);
        }

        // Connected bonds: lines to neighbors, from connected-particles example
        part.neighbors?.forEach(neighId => {
          const neigh = particles.find(p => p.id === neighId);
          if (neigh) {
            p.stroke(255, 100);  // White lines
            p.line(...part.pos, ...neigh.pos);
          }
        });
        p.pop();
      });
    };
  };

  return <p5.Sketch sketch={sketch} />;
