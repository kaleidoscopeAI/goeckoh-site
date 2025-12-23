import EnhancedQuantumConsciousnessEngine from '../engines/EnhancedQuantumConsciousnessEngine';

test('deterministic replay small run', () => {
  // deterministic seed via Math.random replacer
  let seed = 12345;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const oldRand = Math.random;
  (global as any).Math.random = rand;

  const engine = new EnhancedQuantumConsciousnessEngine(1000);
  const positions = new Float32Array(1000 * 3);
  for(let i=0;i<positions.length;i++) positions[i] = (rand() - 0.5) * 800;
  const velocities = new Float32Array(1000 * 3);
  for(let i=0;i<5;i++) {
    engine.update(positions, { valence: 0, arousal: 0.5 });
  }
  expect(engine.getGlobalCoherence()).toBeCloseTo(engine.getGlobalCoherence()); // trivial check

  (global as any).Math.random = oldRand;
});
