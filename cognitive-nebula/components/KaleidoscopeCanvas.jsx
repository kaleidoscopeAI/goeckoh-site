import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { Line, Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Vector3 } from 'three';

// Interactive Node component with tooltip
const Node3D = ({ node, isSelected, isNeighbor, onClick }) => {
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    document.body.style.cursor = hovered ? 'pointer' : 'auto';
  }, [hovered]);

  const scale = isSelected ? 1.5 : isNeighbor ? 1.2 : 1;
  const color = new THREE.Color().setHSL(node.valence, 1, 0.5);
  const emissiveIntensity = isSelected ? 3 : hovered || isNeighbor ? 2 : node.arousal * 1.5;

  return (
    <>
      <mesh
        position={node.position}
        scale={scale}
        onClick={onClick}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          toneMapped={false}
          metalness={0.1}
          roughness={0.5}
        />
      </mesh>
      {isSelected && (
        <Html position={node.position} distanceFactor={12} zIndexRange={[100, 0]} transform>
          <div
            className="bg-gray-900/80 text-white text-xs rounded-md p-2 w-40 border border-cyan-500/50 backdrop-blur-sm select-none"
            style={{ pointerEvents: 'none' }}
          >
            <p className="font-bold text-cyan-400">Node: {node.id}</p>
            <p>Arousal: <span className="text-yellow-300">{node.arousal.toFixed(2)}</span></p>
            <p>Valence: <span className="text-pink-300">{node.valence.toFixed(2)}</span></p>
            <p>Energy: <span className="text-green-300">{node.energy.toFixed(2)}</span></p>
            <p>Awareness: <span className="text-purple-300">{node.awareness.toFixed(2)}</span></p>
          </div>
        </Html>
      )}
    </>
  );
};

// Scene containing all 3D elements
const Scene = ({ nodes, edges, selectedNodeId, setSelectedNodeId }) => {
  const { controls } = useThree();
  const nodeMap = useMemo(() => new Map(nodes.map(n => [n.id, n])), [nodes]);

  const neighbors = useMemo(() => {
    if (!selectedNodeId) return new Set();
    const adjacentNodes = new Set();
    edges.forEach(edge => {
      if (edge.source === selectedNodeId) adjacentNodes.add(edge.target);
      if (edge.target === selectedNodeId) adjacentNodes.add(edge.source);
    });
    return adjacentNodes;
  }, [selectedNodeId, edges]);

  useEffect(() => {
    if (controls) {
      if (selectedNodeId) {
        const node = nodeMap.get(selectedNodeId);
        if (node) {
          const targetPosition = new Vector3(...node.position);
          controls.setTarget(...targetPosition.toArray(), true);
        }
      } else {
        controls.setTarget(0, 0, 0, true);
      }
    }
  }, [selectedNodeId, controls, nodeMap]);

  const handleNodeClick = (e, nodeId) => {
    e.stopPropagation();
    setSelectedNodeId(selectedNodeId === nodeId ? null : nodeId);
  };

  return (
    <group>
      {nodes.map(node => (
        <Node3D
          key={node.id}
          node={node}
          isSelected={selectedNodeId === node.id}
          isNeighbor={neighbors.has(node.id)}
          onClick={(e) => handleNodeClick(e, node.id)}
        />
      ))}
      {edges.map((edge, i) => {
        const sourceNode = nodeMap.get(edge.source);
        const targetNode = nodeMap.get(edge.target);
        if (!sourceNode || !targetNode) return null;

        const isHighlighted = selectedNodeId && (edge.source === selectedNodeId || edge.target === selectedNodeId);

        return (
          <Line
            key={`${edge.source}-${edge.target}-${i}`}
            points={[sourceNode.position, targetNode.position]}
            color={isHighlighted ? "#22d3ee" : "white"}
            lineWidth={isHighlighted ? 1.5 : 0.5}
            transparent
            opacity={isHighlighted ? 0.9 : 0.15}
          />
        );
      })}
    </group>
  );
};

// Main export component
const KaleidoscopeCanvas = ({ nodes, edges, selectedNodeId, setSelectedNodeId }) => {
  return (
    <Canvas
      camera={{ position: [0, 5, 18], fov: 50 }}
      onPointerMissed={(e) => {
        if (e.type === 'click') setSelectedNodeId(null)
      }}
    >
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={1.5} />
      <Scene nodes={nodes} edges={edges} selectedNodeId={selectedNodeId} setSelectedNodeId={setSelectedNodeId} />
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={!selectedNodeId}
        autoRotateSpeed={0.5}
        makeDefault
      />
    </Canvas>
  );
};

export default KaleidoscopeCanvas;
