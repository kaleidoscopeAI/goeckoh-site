import p5 from 'p5';

const App = () => {
  const sketch = (p: p5) => {
    let particles: any[] = [];
    let stars: {x: number, y: number, z: number}[] = [];

    p.setup = () => {
      p.createCanvas(800, 600, p.WEBGL);
      // Starfield from /: random 3D
      for (let i = 0; i < 1000; i++) {
        stars.push({
          x: p.random(-p.width*2, p.width*2),
          y: p.random(-p.height*2, p.height*2),
          z: p.random(p.width)  # Depth
        });
      }
      // WS...
    };

    p.draw = () => {
      p.background(0);

      // Aurora from /: sin wave lerp
      p.noStroke();
      for (let y = -p.height/2; y < p.height/2; y++) {
        const t = p.map(y, -p.height/2, p.height/2, 0, 1) + p.sin(p.frameCount * 0.005 + y * 0.01) * 0.3;  # Wave anim
        const c = p.lerpColor(p.color(0, 50, 100), p.color(120, 50, 100), t);
        p.fill(c);
        p.rect(-p.width/2, y, p.width, 1);
      }

      // Starfield parallax: map to screen, size 1/z, move z--
      p.fill(255);
      stars.forEach(s => {
        const sx = p.map(s.x / s.z, 0, 1, 0, p.width);
        const sy = p.map(s.y / s.z, 0, 1, 0, p.height);
        const size = p.map(s.z, 0, p.width, 4, 0);
        p.ellipse(sx - p.width/2, sy - p.height/2, size, size);
        s.z -= 2;  # Toward viewer
        if (s.z < 1) {  # Reset
          s.x = p.random(-p.width*2, p.width*2);
          s.y = p.random(-p.height*2, p.height*2);
          s.z = p.width;
        }
      });

      particles.forEach((part, i) => {
        // N-body from : accel to all
        let acc = p.createVector(0,0,0);
        particles.forEach((other, j) => {
          if (i !== j) {
            const r = p5.Vector.sub(p.createVector(...other.pos), p.createVector(...part.pos));
            const dist = r.mag() + 1e-9;
            acc.add(r.normalize().mult(-1 / (dist * dist)));  # Gravity
          }
        });
        part.vel = part.vel || p.createVector(0,0,0);
        part.vel.add(acc.mult(0.01));
        part.pos = p5.Vector.add(part.pos, part.vel.mult(0.01));

        p.translate(...part.pos);
        p.fill(part.color);
        p.sphere(part.size);

        // Halo from : blendMode(ADD) multi-sphere glow
        if (part.halo) {
          p.blendMode(p.ADD);
          p.noFill();
          p.stroke(part.color, 100);  # Semi-trans
          p.sphere(part.size * 1.2);
          p.sphere(part.size * 1.5);  # Layers for glow
          p.blendMode(p.BLEND);  # Reset
        }

        // Connected from : lines if dist < thresh
        particles.forEach((other, j) => {
          if (i < j) {  # Avoid double
            const dist = p5.Vector.dist(p.createVector(...part.pos), p.createVector(...other.pos));
            if (dist < 50) {  # Thresh
              const stress = dist / 30;  # Orange if stretched
              p.stroke(`hsl(30, 100%, ${stress*50}%)`);
              p.line(...part.pos, ...other.pos);
            }
          }
        });
      });
    };
  };

  return <p5.Sketch sketch={sketch} />;
