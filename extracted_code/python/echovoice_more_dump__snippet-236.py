import p5 from 'p5';

const App = () => {
  const sketch = (p: p5) => {
    // ... Previous setup/stars/WS

    p.draw = () => {
      p.background(0);

      // Aurora from Claude/YouTube: multi-layer sin lerp transparent
      p.blendMode(p.BLEND);
      const colors = [p.color(0, 255, 255, 50), p.color(0, 255, 0, 50), p.color(255, 0, 255, 50)];  # Cyan/green/purple
      for (let layer = 0; layer < 3; layer++) {
        for (let y = -p.height/2; y < p.height/2; y++) {
          const t = p.map(y, -p.height/2, p.height/2, 0, 1);
          const wave = p.sin(p.frameCount * 0.005 * (layer+1) + y * 0.01 * (layer+1)) * 0.3;  # Multi-sin anim
          const inter = t + wave;
          const c = p.lerpColor(colors[layer % 3], colors[(layer+1) % 3], inter);
          p.stroke(c);
          p.line(-p.width/2, y, p.width/2, y);
        }
      }

      // ... Stars parallax

      particles.forEach((part, i) => {
        // N-body from blog/YouTube/Adesh: calculate_force -G/r^2 * unit
        let acc = p.createVector(0,0,0);
        particles.forEach((other, j) => {
          if (i !== j) {
            const rVec = p5.Vector.sub(p.createVector(...other.pos), p.createVector(...part.pos));
            const r = rVec.mag() + 1e-9;
            const forceMag = -1 / (r * r);  # G=1
            acc.add(rVec.normalize().mult(forceMag));
          }
        });
        part.vel.add(acc.mult(0.01));
        part.pos.add(part.vel.mult(0.01));

        p.translate(...part.pos);
        p.fill(part.color);
        p.sphere(part.size);

        // Halo glow from deconbatch/OpenProcessing: blendMode(ADD) layered
        if (part.halo) {
          p.blendMode(p.ADD);
          p.noFill();
          for (let l = 1; l <= 3; l++) {  # Layers for glow
            p.stroke(part.color, 255 / l);  # Fade alpha
            p.sphere(part.size * (1 + 0.2 * l));  # Expanding
          }
          p.blendMode(p.BLEND);
        }

        // Connected from previous: lines if dist < thresh, orange stress
        // ... 
      });
    };
  };

  return <p5.Sketch sketch={sketch} />;
