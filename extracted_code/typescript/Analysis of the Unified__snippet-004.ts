// src/components/EmbeddedDashboard.tsx
import React from 'react';

interface Props {
  systemState: any;
  particleData: any;
}

export default function EmbeddedDashboard({ systemState, particleData }: Props) {
  return (
    <div className="dashboard">
      <h3>🧠 Embedded Quantum Consciousness</h3>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{systemState.globalCoherence?.toFixed(3) || '0.000'}</div>
          <div className="metric-label">Global Coherence</div>
          <div className="metric-bar">
            <div 
              className="metric-fill" 
              style={{ width: `${(systemState.globalCoherence || 0) * 100}%` }}
            />
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-value">{systemState.knowledgeCrystals || 0}</div>
          <div className="metric-label">Knowledge Crystals</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-value" style={{ 
            color: (systemState.emotionalField?.valence || 0) > 0 ? '#4ade80' : '#f87171' 
          }}>
            {(systemState.emotionalField?.valence || 0).toFixed(2)}
          </div>
          <div className="metric-label">Emotional Valence</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-value">{(systemState.emotionalField?.arousal || 0).toFixed(2)}</div>
          <div className="metric-label">Emotional Arousal</div>
        </div>
      </div>

      {/* Embedded Ollama Status */}
      {systemState.ollamaStatus && (
        <div className="embedded-status">
          <h4>🦙 Embedded Ollama</h4>
          <div className="status-grid">
            <div className="status-item">
              <span className={`status-dot ${systemState.ollamaStatus.running ? 'online' : 'offline'}`}></span>
              Status: {systemState.ollamaStatus.running ? 'Running' : 'Stopped'}
            </div>
            <div className="status-item">
              <span>Queue:</span>
              <span>{systemState.ollamaStatus.queueLength || 0}</span>
            </div>
          </div>
        </div>
      )}

      {/* Dynamic Parameters */}
      {systemState.dynamicParameters && (
        <div className="parameters-section">
          <h4>Ollama-Driven Parameters</h4>
          <div className="parameter-grid">
            <div className="parameter">
              <span>Valence Boost:</span>
              <span>{systemState.dynamicParameters.emotionalValenceBoost?.toFixed(3) || '0.000'}</span>
            </div>
            <div className="parameter">
              <span>Entanglement:</span>
              <span>{systemState.dynamicParameters.quantumEntanglementStrength?.toFixed(3) || '1.000'}</span>
            </div>
            <div className="parameter">
              <span>Mimicry Mod:</span>
              <span>{systemState.dynamicParameters.mimicryForceModifier?.toFixed(3) || '1.000'}</span>
            </div>
          </div>
        </div>
      )}
      
      {/* Hypotheses */}
      <div className="hypotheses-section">
        <h4>
          Active Hypotheses 
          {particleData?.embeddedOllama && <span className="embedded-badge">Embedded</span>}
        </h4>
        {(systemState.hypotheses || []).map((hyp: any, idx: number) => (
          <div key={idx} className={`hypothesis-item ${hyp.analyzed ? 'analyzed' : ''}`}>
            <div className="hypothesis-text">
              {hyp.text}
              {hyp.refined && (
                <div className="refined-hypothesis">
                  <strong>Refined:</strong> {hyp.refined}
                </div>
              )}
            </div>
            <div className="hypothesis-meta">
              <span className="confidence">
                Confidence: {((hyp.confidence || 0) * 100).toFixed(1)}%
              </span>
              {hyp.plausibility && (
                <span className="plausibility">
                  Plausibility: {(hyp.plausibility * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
