import { useState, useEffect } from 'react';

const BASE_URL = 'http://localhost:5000';

export const useNodeSimulation = (nodeCount) => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [metrics, setMetrics] = useState({ nodeCount, updateTime: 0 });

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
      }
    };

    startSimulation();
  }, [nodeCount]);

  useEffect(() => {
    const updateSimulation = async () => {
      const startTime = performance.now();
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
      }
      const endTime = performance.now();
      setMetrics(prevMetrics => ({ ...prevMetrics, updateTime: endTime - startTime }));
    };

    const intervalId = setInterval(updateSimulation, 1000); // Update every second

    return () => clearInterval(intervalId);
  }, []);

  return { nodes, edges, metrics };
};