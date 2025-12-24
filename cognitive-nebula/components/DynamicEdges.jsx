import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const DynamicEdges = ({ nodes, edges, selectedNode, hoveredNodeId }) => {
  const normalRef = useRef();
  const highlightRef = useRef();

  const maxEdges = edges.length;
  const normalPositions = useMemo(() => new Float32Array(maxEdges * 6), [maxEdges]);
  const highlightPositions = useMemo(() => new Float32Array(maxEdges * 6), [maxEdges]);

  const normalGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(normalPositions, 3));
    return geo;
  }, [normalPositions]);

  const highlightGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(highlightPositions, 3));
    return geo;
  }, [highlightPositions]);

  const normalMaterial = useMemo(() => new THREE.LineBasicMaterial({
    color: '#ffffff',
    transparent: true,
    opacity: 0.15,
  }), []);

  const highlightMaterial = useMemo(() => new THREE.LineBasicMaterial({
    color: '#22d3ee',
    transparent: true,
    opacity: 0.9,
  }), []);

  useFrame(() => {
    const selectedId = selectedNode?.id ?? -1;
    const hoveredId = hoveredNodeId ?? -1;

    let normalCount = 0;
    let highlightCount = 0;

    for (let i = 0; i < edges.length; i++) {
      const { source, target } = edges[i];
      const srcNode = nodes[source];
      const tgtNode = nodes[target];
      if (!srcNode || !tgtNode) continue;

      const isHighlighted = source === selectedId || target === selectedId || source === hoveredId || target === hoveredId;

      const offset = (isHighlighted ? highlightCount : normalCount) * 6;
      const positionsArray = isHighlighted ? highlightPositions : normalPositions;

      positionsArray[offset] = srcNode.pos[0];
      positionsArray[offset + 1] = srcNode.pos[1];
      positionsArray[offset + 2] = srcNode.pos[2];
      positionsArray[offset + 3] = tgtNode.pos[0];
      positionsArray[offset + 4] = tgtNode.pos[1];
      positionsArray[offset + 5] = tgtNode.pos[2];

      if (isHighlighted) highlightCount++;
      else normalCount++;
    }

    normalGeometry.setDrawRange(0, normalCount * 2);
    highlightGeometry.setDrawRange(0, highlightCount * 2);

    normalGeometry.attributes.position.needsUpdate = true;
    highlightGeometry.attributes.position.needsUpdate = true;
  });

  return (
    <>
      <lineSegments ref={normalRef} geometry={normalGeometry} material={normalMaterial} />
      <lineSegments ref={highlightRef} geometry={highlightGeometry} material={highlightMaterial} />
    </>
  );
};

export default DynamicEdges;
