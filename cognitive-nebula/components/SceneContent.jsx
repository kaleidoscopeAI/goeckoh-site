import React, { useRef, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

import InstancedNodes from './InstancedNodes.jsx';
import DynamicEdges from './DynamicEdges.jsx';

export default function SceneContent({ nodes, edges, selectedNode, hoveredNodeId, onNodeHover, onNodeClick }) {
  const controlsRef = useRef();
  const { camera } = useThree();

  useEffect(() => {
    if (controlsRef.current) {
      if (selectedNode) {
        controlsRef.current.target.set(...selectedNode.pos);
        controlsRef.current.update();
        camera.lookAt(new THREE.Vector3(...selectedNode.pos));
      } else {
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
      }
    }
  }, [selectedNode, camera]);

  return (
    <>
      <color attach="background" args={['#05050a']} />
      <ambientLight intensity={0.5} />
      <pointLight position={[100, 100, 100]} intensity={1.5} color="#88aaff" />
      <pointLight position={[-100, -100, -100]} intensity={1.5} color="#ff88aa" />

      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

      {nodes && nodes.length > 0 && (
        <>
          <InstancedNodes
            nodes={nodes}
            instanceCap={5000} // Assuming NODE_COUNT from App.jsx
            hoveredNodeId={hoveredNodeId}
            onNodeHover={onNodeHover}
            onNodeClick={onNodeClick}
          />
          <DynamicEdges
            nodes={nodes}
            edges={edges}
            selectedNode={selectedNode}
            hoveredNodeId={hoveredNodeId}
          />
        </>
      )}

      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.5}
        minDistance={50}
        maxDistance={500}
        autoRotate={!selectedNode}
        autoRotateSpeed={0.5}
      />
    </>
  );
}
