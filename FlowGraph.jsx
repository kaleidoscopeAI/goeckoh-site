import React from 'react';

const FlowGraph = ({ flowData }) => {
  return (
    <div className="absolute bottom-4 right-4 bg-gray-800/50 p-4 rounded-lg text-white font-mono text-sm">
      <h2 className="font-bold text-lg mb-2">Data Flow</h2>
      {/* Placeholder for a more complex graph */}
      <p>{flowData}</p>
    </div>
  );
};

export default FlowGraph;
