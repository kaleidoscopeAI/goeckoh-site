import { useEffect, useRef, useState } from 'react';

const BASE_URL = 'http://localhost:5000';

export const useNodeSimulation = (nodeCount) => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [metrics, setMetrics] = useState({ nodeCount, updateTime: 0 });
  const syntheticRef = useRef(false);
  const tickRef = useRef(0);

  const createSyntheticState = () => {
    const syntheticNodes = Array.from({ length: Math.min(nodeCount, 800) }, (_, i) => {
      const phase = i / Math.min(nodeCount, 800);
      const radius = 40 + (phase * 40);
      const angle = phase * Math.PI * 4;
      return {
        id: i,
        pos: [
          Math.cos(angle) * radius,
          Math.sin(angle * 1.3) * radius * 0.6,
          Math.sin(angle) * radius * 0.8,
        ],
        vel: [0, 0, 0],
        E: 0.4 + Math.random() * 0.6,
        A: 0.2 + Math.random() * 0.5,
        K: (i % 360) / 360,
      };
    });

    const syntheticEdges = syntheticNodes.flatMap((node, idx) => {
      const links = [];
      for (let j = 1; j <= 3; j++) {
        const target = (idx + j) % syntheticNodes.length;
        links.push({ source: node.id, target });
      }
      return links;
    });

    return { syntheticNodes, syntheticEdges };
  };

  const advanceSynthetic = () => {
    tickRef.current += 1;
    setNodes((prev) =>
      prev.map((node, idx) => {
        const phase = (tickRef.current * 0.02) + idx * 0.03;
        const wobble = Math.sin(phase) * 4;
        return {
          ...node,
          pos: [
            node.pos[0] * Math.cos(0.002) - node.pos[2] * Math.sin(0.002),
            node.pos[1] + wobble * 0.1,
            node.pos[0] * Math.sin(0.002) + node.pos[2] * Math.cos(0.002),
          ],
          E: 0.5 + 0.4 * Math.sin(phase + idx * 0.01),
          A: 0.5 + 0.25 * Math.cos(phase * 0.7),
        };
      })
    );
    setMetrics((prev) => ({ ...prev, updateTime: 16 + Math.random() * 4 }));
  };

  useEffect(() => {
    const startSimulation = async () => {
      try {
        const response = await fetch(`${BASE_URL}/api/simulation/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ data_volume: nodeCount * 10, memory_per_node: 10 }),
        });
        const data = await response.json();
        console.log('Simulation started:', data);
        fetchSimulationState();
      } catch (error) {
        console.error('Error starting simulation:', error);
        const { syntheticNodes, syntheticEdges } = createSyntheticState();
        syntheticRef.current = true;
        setNodes(syntheticNodes);
        setEdges(syntheticEdges);
        setMetrics({ nodeCount: syntheticNodes.length, updateTime: 0 });
      }
    };

    const fetchSimulationState = async () => {
      try {
        const response = await fetch(`${BASE_URL}/api/simulation/state`);
        const data = await response.json();
        if (data.nodes) {
          setNodes(data.nodes.map(node => ({...node, pos: node.position})));
          // Assuming edges can be derived from bonds or are static for now
          // This part might need adjustment based on backend data
          const newEdges = [];
          data.nodes.forEach(node => {
            if (node.bonds) {
              node.bonds.forEach(bond => {
                newEdges.push({ source: node.id, target: bond });
              });
            }
          });
          setEdges(newEdges);
          setMetrics(data.metrics);
        }
      } catch (error) {
        console.error('Error fetching simulation state:', error);
        if (!syntheticRef.current) {
          const { syntheticNodes, syntheticEdges } = createSyntheticState();
          syntheticRef.current = true;
          setNodes(syntheticNodes);
          setEdges(syntheticEdges);
          setMetrics({ nodeCount: syntheticNodes.length, updateTime: 0 });
        }
      }
    };

    startSimulation();
  }, [nodeCount]);

  useEffect(() => {
    const updateSimulation = async () => {
      const startTime = performance.now();
      if (syntheticRef.current) {
        advanceSynthetic();
        const endSynthetic = performance.now();
        setMetrics((prev) => ({ ...prev, updateTime: endSynthetic - startTime }));
        return;
      }
      try {
        const response = await fetch(`${BASE_URL}/api/simulation/step`, { method: 'POST' });
        const data = await response.json();
        if (data.nodes) {
          setNodes(data.nodes.map(node => ({...node, pos: node.position})));
          const newEdges = [];
          data.nodes.forEach(node => {
            if (node.bonds) {
              node.bonds.forEach(bond => {
                newEdges.push({ source: node.id, target: bond });
              });
            }
          });
          setEdges(newEdges);
          setMetrics(data.metrics);
        }
      } catch (error) {
        console.error('Error stepping simulation:', error);
        advanceSynthetic();
      }
      const endTime = performance.now();
      setMetrics(prevMetrics => ({ ...prevMetrics, updateTime: endTime - startTime }));
    };

    const intervalId = setInterval(updateSimulation, 1000); // Update every second

    return () => clearInterval(intervalId);
  }, []);

  return { nodes, edges, metrics };
};
