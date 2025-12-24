import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import ShaderMaterial from './ShaderMaterial';

const InstancedNodes = ({
  nodes,
  instanceCap,
  hoveredNodeId,
  onNodeHover,
  onNodeClick,
}) => {
  const meshRef = useRef();
  const { camera, gl, raycaster, mouse } = useThree();

  const colorsRef = useRef(new THREE.InstancedBufferAttribute(new Float32Array(instanceCap * 3), 3));
  const intensitiesRef = useRef(new THREE.InstancedBufferAttribute(new Float32Array(instanceCap), 1));

  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(1.5, 8, 8);
    geo.setAttribute('instanceColor', colorsRef.current);
    geo.setAttribute('instanceIntensity', intensitiesRef.current);
    return geo;
  }, [instanceCap]);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  // Performance: Reuse a single Color object to avoid allocations on each frame
  const tempColor = useMemo(() => new THREE.Color(), []);

  useFrame(() => {
    if (!meshRef.current) return;

    nodes.forEach((node, i) => {
      const isHovered = node.id === hoveredNodeId;
      const scale = isHovered ? 2 : 1 + node.A;

      dummy.position.set(...node.pos);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);

      // Performance: Reuse tempColor instead of creating new Color() each iteration
      tempColor.setHSL(node.K, 1.0, 0.5);
      colorsRef.current.setXYZ(i, tempColor.r, tempColor.g, tempColor.b);
      intensitiesRef.current.setX(i, node.E * 3 + (isHovered ? 2 : 0));
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    colorsRef.current.needsUpdate = true;
    intensitiesRef.current.needsUpdate = true;
  });

  useEffect(() => {
    const handlePointerMove = (event) => {
      mouse.x = (event.clientX / gl.domElement.clientWidth) * 2 - 1;
      mouse.y = -(event.clientY / gl.domElement.clientHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef.current, true);
      if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
        onNodeHover(nodes[intersects[0].instanceId].id);
      } else {
        onNodeHover(null);
      }
    };

    const handleClick = (event) => {
      mouse.x = (event.clientX / gl.domElement.clientWidth) * 2 - 1;
      mouse.y = -(event.clientY / gl.domElement.clientHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef.current, true);
      if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
        onNodeClick(nodes[intersects[0].instanceId]);
      } else {
        onNodeClick(null);
      }
    };

    gl.domElement.addEventListener('pointermove', handlePointerMove);
    gl.domElement.addEventListener('click', handleClick);

    return () => {
      gl.domElement.removeEventListener('pointermove', handlePointerMove);
      gl.domElement.removeEventListener('click', handleClick);
    };
  }, [nodes, camera, gl, raycaster, mouse, onNodeHover, onNodeClick]);

  return <instancedMesh ref={meshRef} args={[geometry, ShaderMaterial(), instanceCap]} />;
};

export default InstancedNodes;
