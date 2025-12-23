  - Words from the SSE transcript now map to themed colors and simple shapes
    (sphere, cube, cone, torus, pyramid) based on keywords like “dog”, “car”,
    “tree”, etc. Metadata is attached per node so you can branch on shape in
    your Three.js rendering.
  - This keeps the stream immediate (no extra backend calls) and makes phrases
    like “dog chasing car” visually distinct.

  Files touched:

  - cognitive-nebula/agi-thought-visualizer/App.tsx (color/geometry mapping
    added)

  If you want the shapes actually rendered (e.g., swapping particle sprites for
  meshes based on metadata.shape), I can update Visualization.tsx next to
  instantiate the corresponding geometries.


