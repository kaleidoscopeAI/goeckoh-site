import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function UnifiedCognitiveSystem() {
  const mountRef = useRef(null);
  const [systemState, setSystemState] = useState(null);
  const [lastSummary, setLastSummary] = useState("Connecting to UNI Core...");

  // This holds the Three.js objects that correspond to our AI nodes
  const visualNodesRef = useRef({});

  // Function to trigger a cognitive cycle in the backend
  async function triggerSpeculation(input) {
    try {
      const response = await axios.post(`${API_URL}/speculate`, {
        system_metrics: { userInput: input }, // Sending input as part of metrics
        adc_raw: [Math.random(), Math.random()], // Mock sensor data
      });
      setSystemState(response.data);
    } catch (err) {
      console.error('Speculation error:', err);
      setLastSummary("Error triggering speculation.");
    }
  }

  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    const group = new THREE.Group();
    scene.add(group);

    const camera = new THREE.PerspectiveCamera(75, mount.clientWidth / mount.clientHeight, 0.1, 5000);
    camera.position.z = 200;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // --- Real-time data fetching loop ---
    const fetchDataInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/status`);
        setSystemState(response.data);
        if(response.data.llm_summary) {
            setLastSummary(response.data.llm_summary);
        }
      } catch (err) {
        console.error('Status fetch error:', err);
        setLastSummary("Connection to UNI Core lost.");
      }
    }, 200); // Fetch state 5 times per second

    // --- Animation loop ---
    function animate() {
      if (systemState && systemState.nodes) {
        // If the number of nodes in backend changes, update the scene
        if (Object.keys(visualNodesRef.current).length !== systemState.nodes.length) {
            // Clear existing nodes
            group.clear();
            visualNodesRef.current = {};
            
            const geometry = new THREE.SphereGeometry(2, 12, 12);
            systemState.nodes.forEach(node => {
                const material = new THREE.MeshBasicMaterial({ color: 0x00ffff });
                const sphere = new THREE.Mesh(geometry, material);
                visualNodesRef.current[node.id] = sphere;
                group.add(sphere);
            });
        }

        // Update positions and colors of existing nodes
        systemState.nodes.forEach(node => {
          const visualNode = visualNodesRef.current[node.id];
          if (visualNode) {
            const targetPosition = new THREE.Vector3(...node.pos);
            visualNode.position.lerp(targetPosition, 0.1);
            
            // Color based on arousal (brightness) and valence (hue)
            const hue = (node.valence + 1) / 2; // Map valence [-1, 1] to hue [0, 1]
            const lightness = 0.4 + node.arousal * 0.6; // Map arousal [0, 1] to lightness
            visualNode.material.color.setHSL(hue, 1.0, lightness);
          }
        });
      }

      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }
    animate();

    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      mount.removeChild(renderer.domElement);
      window.removeEventListener('resize', handleResize);
      clearInterval(fetchDataInterval);
    };
  }, [systemState]); // Re-run effect if systemState structure changes fundamentally

  return (
    <div className="w-full h-screen bg-black relative overflow-hidden">
      <div ref={mountRef} className="absolute inset-0" />

      <div className="absolute top-4 left-4 bg-black/60 p-3 rounded-lg text-cyan-300 text-sm space-y-1 max-w-md">
        <div>🧠 AI Thought: {lastSummary}</div>
        {systemState && (
            <>
                <div>🌀 System Purity: {systemState.system_purity?.toFixed(3)}</div>
                <div>❤️ Emotional Valence: {systemState.emotional_valence?.toFixed(3)}</div>
                <div>Nodes: {systemState.node_count}</div>
            </>
        )}
      </div>

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-3/4 max-w-2xl bg-black/60 rounded-lg p-4">
        <input
          type="text"
          placeholder="Provide sensory input to the AI..."
          className="w-full bg-transparent text-white border border-cyan-400 rounded p-2"
          onKeyDown={async (e) => {
            if (e.key === 'Enter') {
              const query = e.target.value.trim();
              if (query.length > 0) {
                await triggerSpeculation(query);
                e.target.value = '';
              }
            }
          }}
        />
      </div>
    </div>
  );
}