import React, { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';

// Import real components and hooks
import HUD from './HUD.jsx';
import { useNodeSimulation } from './simulationService.js';
import SceneContent from './SceneContent.jsx';
import JacksonCompanion from './JacksonCompanion.jsx';

const NODE_COUNT = 5000;

export default function App() {
  const { nodes, edges, metrics } = useNodeSimulation(NODE_COUNT);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [mode, setMode] = useState('lattice'); // 'lattice' | 'companion'

  const containerStyle = {
    width: '100%',
    minHeight: '70vh',
    height: '70vh',
    background: 'radial-gradient(circle at 20% 20%, rgba(34,211,238,0.08), transparent 35%), #030712',
    color: '#e5e7eb',
    position: 'relative',
    borderRadius: '16px',
    border: '1px solid rgba(45,212,191,0.2)',
    boxShadow: '0 25px 80px rgba(0,0,0,0.35)',
    overflow: 'hidden',
  };

  const switcherStyle = {
    position: 'absolute',
    top: '14px',
    left: '14px',
    zIndex: 10,
    display: 'flex',
    gap: '10px',
  };

  const buttonStyle = (active) => ({
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid rgba(56,189,248,0.4)',
    background: active ? 'rgba(8,47,73,0.8)' : 'rgba(15,23,42,0.7)',
    color: '#e0f2fe',
    cursor: 'pointer',
    fontWeight: 600,
    letterSpacing: '0.01em',
  });

  return (
    <div style={containerStyle}>
      <div style={switcherStyle}>
        <button
          onClick={() => setMode('lattice')}
          style={buttonStyle(mode === 'lattice')}
        >
          Lattice View
        </button>
        <button
          onClick={() => setMode('companion')}
          style={buttonStyle(mode === 'companion')}
        >
          Voice Companion
        </button>
      </div>

      {mode === 'companion' ? (
        <JacksonCompanion />
      ) : (
        <>
          <HUD metrics={metrics} selectedNode={selectedNode} />
          <Canvas camera={{ position: [0, 0, 150], fov: 75 }} dpr={[1, 2]} style={{ width: '100%', height: '100%' }}>
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
