for (const n of this.nodes.values()) {
  // Real bit flow: L0 XOR, L3 mat mul (mathjs)
  n.e = n.e.map(b => b ^ (Math.random() < 0.01 ? 1 : 0));  # L0 mutate
  const bitMat = math.matrix([n.e.slice(0,64), n.e.slice(64)]);  # L3 mat
  const mul = math.multiply(bitMat, math.matrix([[1,0],[0,1]]));  # Transform
  n.e = mul.toArray().flat();  # Propagate changes
  // ... Up to L19: E = math.subtract(n.energy, math.multiply(n.a, math.cos(deltaPhi)))  # Physics E
}
