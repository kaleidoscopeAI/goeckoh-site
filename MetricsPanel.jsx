import React from 'react';

const MetricsPanel = ({ metrics }) => {
  return (
    <div className="absolute top-4 right-4 bg-gray-800/50 p-4 rounded-lg text-white font-mono text-sm">
      <h2 className="font-bold text-lg mb-2">System Metrics</h2>
      <p>Node Count: {metrics.nodeCount}</p>
      <p>Update Time: {metrics.updateTime.toFixed(2)} ms</p>
    </div>
  );
};

export default MetricsPanel;
