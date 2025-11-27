import React from 'react';

const HUD = ({ metrics, selectedNode }) => {
  return (
    <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm p-4 rounded-lg text-white font-mono text-sm space-y-2 shadow-lg border border-cyan-500/20 max-w-md">
      <h2 className="font-bold text-cyan-300 text-base border-b border-cyan-500/30 pb-1 mb-2">Simulation Metrics</h2>
      <div className="flex items-center space-x-2">
        <span className="text-cyan-400 w-24 flex-shrink-0">Nodes:</span> 
        <span className="flex-1">{metrics.nodeCount}</span>
      </div>
      <div className="flex items-center space-x-2">
        <span className="text-cyan-400 w-24 flex-shrink-0">Update Time:</span>
        <span className="flex-1">{metrics.updateTime.toFixed(2)} ms</span>
      </div>
      {selectedNode && (
        <div className="mt-4">
          <h3 className="font-bold text-cyan-300 text-base border-b border-cyan-500/30 pb-1 mb-2">Selected Node {selectedNode.id}</h3>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">Position:</span> 
            <span className="flex-1">{selectedNode.pos.map(p => p.toFixed(2)).join(', ')}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">Velocity:</span> 
            <span className="flex-1">{selectedNode.vel.map(v => v.toFixed(2)).join(', ')}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">Energy:</span> 
            <span className="flex-1">{selectedNode.E.toFixed(2)}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">Attention:</span> 
            <span className="flex-1">{selectedNode.A.toFixed(2)}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">Knowledge:</span> 
            <span className="flex-1">{selectedNode.K.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default HUD;
