import React from 'react';

const DrugPanel = ({ drugDiscovery }) => {
  return (
    <div className="absolute bottom-4 right-4 bg-gray-800/50 p-4 rounded-lg text-white font-mono text-sm">
      <h2 className="font-bold text-lg mb-2">Drug Discovery</h2>
      <p>Status: {drugDiscovery.status}</p>
      <p>Current Target: {drugDiscovery.currentTarget}</p>
      <p>Top Candidate: {drugDiscovery.topCandidate}</p>
    </div>
  );
};

export default DrugPanel;
