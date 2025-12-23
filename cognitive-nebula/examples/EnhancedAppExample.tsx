/**
 * Example: Enhanced App with Voice-Reactive Visuals
 * 
 * This example shows how to integrate the enhanced visual components
 * with your voice therapy system.
 */

import React, { useState, useEffect } from 'react';
import EnhancedThreeCanvas from '../components/EnhancedThreeCanvas';
import EnhancedBubble from '../components/EnhancedBubble';
import VoiceVisualizer from '../components/VoiceVisualizer';
import type { Metrics, Settings } from '../types';

// Example: Mock voice data (replace with real data from your voice pipeline)
interface VoiceData {
  bubbleState: {
    radius: number;
    color_r: number;
    color_g: number;
    color_b: number;
    rough: number;
    metal: number;
    spike: number;
    energy: number;
    f0: number;
  };
  waveform?: Float32Array;
  spectrum?: Float32Array;
}

const EnhancedAppExample: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics>({
    curiosity: 0.7,
    confusion: 0.3,
    coherence: 0.5,
    arousal: 0.4,
    valence: 0.6,
    dominance: 0.5,
    certainty: 0.5,
    resonance: 0.6,
  });

  const [settings, setSettings] = useState<Settings>({
    particleCount: 0.5,
    colorSaturation: 1.0,
    movementSpeed: 1.0,
    showStars: true,
    showTrails: true,
    temperature: 1.1,
    topP: 0.9,
    presencePenalty: 0.4,
    stylePreset: 'rotate',
    sdHost: 'http://localhost:7860',
    sdSteps: 24,
    sdCfgScale: 7.5,
    sdSampler: 'Euler a',
    sdSeed: null,
  });

  const [voiceData, setVoiceData] = useState<VoiceData | undefined>(undefined);
  const [showBubble, setShowBubble] = useState(true);
  const [showVisualizer, setShowVisualizer] = useState(false);

  // Example: Simulate voice data updates
  useEffect(() => {
    const interval = setInterval(() => {
      // In real implementation, this would come from your voice pipeline
      setVoiceData({
        bubbleState: {
          radius: 1.0 + Math.random() * 0.5,
          color_r: 0.25 + Math.random() * 0.2,
          color_g: 0.7 + Math.random() * 0.2,
          color_b: 1.0,
          rough: 0.3 + Math.random() * 0.3,
          metal: 0.5 + Math.random() * 0.3,
          spike: Math.random() * 0.5,
          energy: Math.random(),
          f0: 200 + Math.random() * 100,
        },
        // waveform: new Float32Array(512).map(() => Math.random() * 2 - 1),
        // spectrum: new Float32Array(128).map(() => Math.random()),
      });
    }, 100); // Update 10 times per second

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden">
      {/* Main Nebula Visualization */}
      <EnhancedThreeCanvas
        metrics={metrics}
        aiThought="Voice-reactive therapeutic visualization"
        imageData={null}
        settings={settings}
        voiceData={voiceData}
      />

      {/* Enhanced Bubble (overlay) */}
      {showBubble && voiceData && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2">
            <EnhancedBubble
              bubbleState={voiceData.bubbleState}
              voiceData={voiceData}
              settings={{
                enableBloom: true,
                enableParticles: true,
                enableVolumetric: true,
                particleCount: 3000,
              }}
            />
          </div>
        </div>
      )}

      {/* Voice Visualizer (optional overlay) */}
      {showVisualizer && voiceData && (
        <div className="absolute bottom-0 right-0 w-1/3 h-1/3 pointer-events-none">
          <VoiceVisualizer
            waveform={voiceData.waveform}
            spectrum={voiceData.spectrum}
            f0={voiceData.bubbleState.f0}
            energy={voiceData.bubbleState.energy}
            color={new THREE.Color(
              voiceData.bubbleState.color_r,
              voiceData.bubbleState.color_g,
              voiceData.bubbleState.color_b
            )}
          />
        </div>
      )}

      {/* Control Panel */}
      <div className="absolute top-4 left-4 bg-black/50 p-4 rounded-lg backdrop-blur-sm">
        <h3 className="text-white mb-2">Visual Controls</h3>
        <div className="space-y-2">
          <label className="flex items-center text-white">
            <input
              type="checkbox"
              checked={showBubble}
              onChange={(e) => setShowBubble(e.target.checked)}
              className="mr-2"
            />
            Show Bubble
          </label>
          <label className="flex items-center text-white">
            <input
              type="checkbox"
              checked={showVisualizer}
              onChange={(e) => setShowVisualizer(e.target.checked)}
              className="mr-2"
            />
            Show Voice Visualizer
          </label>
        </div>
      </div>

      {/* Voice Data Display */}
      {voiceData && (
        <div className="absolute bottom-4 left-4 bg-black/50 p-4 rounded-lg backdrop-blur-sm text-white text-sm">
          <div>Energy: {(voiceData.bubbleState.energy * 100).toFixed(0)}%</div>
          <div>F0: {voiceData.bubbleState.f0.toFixed(0)} Hz</div>
          <div>Spike: {(voiceData.bubbleState.spike * 100).toFixed(0)}%</div>
          <div>Radius: {voiceData.bubbleState.radius.toFixed(2)}</div>
        </div>
      )}
    </div>
  );
};

export default EnhancedAppExample;

