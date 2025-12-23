import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import axios from 'axios';

const MolecularCube = () => {
  const mountRef = useRef(null);
  const [molecules, setMolecules] = useState();
  const [smiles1, setSmiles1] = useState('');
  const [smiles2, setSmiles2] = useState('');
  const [similarity, setSimilarity] = useState(null);

  useEffect(() => {
    // Fetch initial molecule data (replace with real data loading)
    const loadMolecules = async () => {
      // Placeholder - replace with API call or data loading
      const mockMolecules = [
        { smiles: 'CC(=O)Oc1ccccc1C(=O)O' }, // Aspirin
        { smiles: 'c1ccccc1' }, // Benzene
        //... more molecules
      ];
      setMolecules(mockMolecules);
    };

    loadMolecules();

    // Three.js Scene Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    // Add lights (important for good 3D rendering)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);


    // Create 3D objects (spheres - simplified)
    molecules.forEach(molecule => {
        const geometry = new THREE.SphereGeometry(0.1, 32, 32);
        const material = new THREE.MeshStandardMaterial({ color: 0xffa500 }); // Use MeshStandardMaterial for lighting
        const sphere = new THREE.Mesh(geometry, material);

        // Placeholder positions (replace with data from API)
        sphere.position.x = Math.random() * 10 - 5;
        sphere.position.y = Math.random() * 10 - 5;
        sphere.position.z = Math.random() * 10 - 5;

        scene.add(sphere);
    });

    camera.position.z = 5;

    const animate = function () {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };

    animate();

    return () => {
        mountRef.current.removeChild(renderer.domElement);
    };
  }, [molecules]); // Re-run effect when molecules change

  const calculateSimilarity = async () => {
    try {
      const response = await axios.post('/api/similarity', { smiles1, smiles2 });
      const data = await response.json();
      setSimilarity(data.similarity);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      {/*... Input fields and buttons... */}
      <div ref={mountRef} style={{ height: '600px' }} /> {/* 3D visualization */}
      {/*... Similarity results... */}
    </div>
  );
