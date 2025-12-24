import React from 'react';

const HUD = ({ metrics, selectedNode }) => {
  const panelStyle = {
    position: 'absolute',
    top: '18px',
    right: '18px',
    background: 'rgba(3,7,18,0.78)',
    backdropFilter: 'blur(8px)',
    padding: '14px',
    borderRadius: '12px',
    color: '#e2e8f0',
    fontFamily: 'JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '0.85rem',
    boxShadow: '0 15px 60px rgba(0,0,0,0.35)',
    border: '1px solid rgba(45,212,191,0.25)',
    maxWidth: '320px',
  };

  const rowStyle = { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' };
  const labelStyle = { color: '#67e8f9', width: '110px', flexShrink: 0 };

  return (
    <div style={panelStyle}>
      <h2 style={{ fontWeight: 800, color: '#67e8f9', fontSize: '1rem', marginBottom: '10px', borderBottom: '1px solid rgba(45,212,191,0.35)', paddingBottom: '6px' }}>
        Simulation Metrics
      </h2>
      <div style={rowStyle}>
        <span style={labelStyle}>Nodes:</span> 
        <span style={{ flex: 1 }}>{metrics.nodeCount}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>Update Time:</span>
        <span style={{ flex: 1 }}>{metrics.updateTime.toFixed(2)} ms</span>
      </div>
      {selectedNode && (
        <div style={{ marginTop: '10px' }}>
          <h3 style={{ fontWeight: 800, color: '#67e8f9', fontSize: '1rem', marginBottom: '10px', borderBottom: '1px solid rgba(45,212,191,0.35)', paddingBottom: '6px' }}>
            Selected Node {selectedNode.id}
          </h3>
          <div style={rowStyle}>
            <span style={labelStyle}>Position:</span> 
            <span style={{ flex: 1 }}>{selectedNode.pos.map(p => p.toFixed(2)).join(', ')}</span>
          </div>
          <div style={rowStyle}>
            <span style={labelStyle}>Velocity:</span> 
            <span style={{ flex: 1 }}>{selectedNode.vel.map(v => v.toFixed(2)).join(', ')}</span>
          </div>
          <div style={rowStyle}>
            <span style={labelStyle}>Energy:</span> 
            <span style={{ flex: 1 }}>{selectedNode.E.toFixed(2)}</span>
          </div>
          <div style={rowStyle}>
            <span style={labelStyle}>Attention:</span> 
            <span style={{ flex: 1 }}>{selectedNode.A.toFixed(2)}</span>
          </div>
          <div style={rowStyle}>
            <span style={labelStyle}>Knowledge:</span> 
            <span style={{ flex: 1 }}>{selectedNode.K.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default HUD;
