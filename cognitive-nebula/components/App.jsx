import React, { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';

// Import real components and hooks
import HUD from './components/HUD.jsx';
import { useNodeSimulation } from './services/simulationService.js';
import SceneContent from './components/SceneContent.jsx';
import JacksonCompanion from './JacksonCompanion.jsx';

const NODE_COUNT = 5000;

export default function App() {
  const { nodes, edges, metrics } = useNodeSimulation(NODE_COUNT);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [mode, setMode] = useState('lattice'); // 'lattice' | 'companion'

  return (
    <div className="w-screen h-screen bg-black text-white">
      <div className="absolute top-2 left-2 z-10 flex gap-2">
        <button
          onClick={() => setMode('lattice')}
          className={`px-3 py-1 rounded ${mode === 'lattice' ? 'bg-cyan-700' : 'bg-gray-800'}`}
        >
          Lattice View
        </button>
        <button
          onClick={() => setMode('companion')}
          className={`px-3 py-1 rounded ${mode === 'companion' ? 'bg-cyan-700' : 'bg-gray-800'}`}
        >
          Voice Companion
        </button>
      </div>

      {mode === 'companion' ? (
        <JacksonCompanion />
      ) : (
        <>
          <HUD metrics={metrics} selectedNode={selectedNode} />
          <Canvas camera={{ position: [0, 0, 150], fov: 75 }} dpr={[1, 2]}>
            <SceneContent
              nodes={nodes}
              edges={edges}
              selectedNode={selectedNode}
              hoveredNodeId={hoveredNodeId}
              onNodeHover={setHoveredNodeId}
              onNodeClick={setSelectedNode}
            />
          </Canvas>
        </>
      )}
    </div>
  );
}
