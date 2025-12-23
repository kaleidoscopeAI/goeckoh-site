import React, { useState, useEffect } from 'react';
import * as THREE from 'three';
import axios from 'axios';

const MolecularCube = () => {
  const mountRef = useRef(null);
  const [molecules, setMolecules] = useState();
  const [similarityResult, setSimilarityResult] = useState(null);
  const [smiles1, setSmiles1] = useState('');
  const [smiles2, setSmiles2] = useState('');

  useEffect(() => {
    // Fetch molecules from the API
    const fetchMolecules = async () => {
      try {
        const response = await axios.get('/api/molecules');
        setMolecules(response.data);
      } catch (error) {
        console.error("Error fetching molecules:", error);
      }
    };

    fetchMolecules();

    // Three.js setup (simplified)
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    // Add lights
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

  const handleSimilaritySearch = async () => {
    try {
      const response = await axios.post('/api/similarity', { smiles1, smiles2 });
      const taskId = response.data.task_id;

      // Poll for the result (not ideal, but a simple example)
      const checkResult = async () => {
        const resultResponse = await axios.get(`/api/similarity_result/${taskId}`);
        if (resultResponse.data.status === 'SUCCESS') {
          setSimilarityResult(resultResponse.data.result);
        } else {
          setTimeout(checkResult, 1000); // Check again after 1 second
        }
      };

      checkResult();
    } catch (error) {
      console.error("Error searching for similarity:", error);
    }
  };

  return (
    <div>
      <div ref={mountRef} style={{ height: '600px' }} />
      <div>
        <input type="text" placeholder="SMILES 1" value={smiles1} onChange={e => setSmiles1(e.target.value)} />
        <input type="text" placeholder="SMI

