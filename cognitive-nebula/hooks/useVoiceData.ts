import { useState, useEffect, useCallback } from 'react';

interface VoiceMetrics {
  energy: number;
  f0: number;
  clarity: number;
  fluency: number;
  volume: number;
  pitchStability: number;
}

interface VoiceData {
  metrics: VoiceMetrics;
  waveform?: Float32Array;
  spectrum?: Float32Array;
  timestamp: number;
}

interface UseVoiceDataOptions {
  updateInterval?: number;
  smoothing?: number;
  onUpdate?: (data: VoiceData) => void;
}

/**
 * Custom hook for managing voice data
 * Connects to voice pipeline and provides real-time updates
 */
export const useVoiceData = (options: UseVoiceDataOptions = {}) => {
  const {
    updateInterval = 100,
    smoothing = 0.1,
    onUpdate
  } = options;

  const [voiceData, setVoiceData] = useState<VoiceData>({
    metrics: {
      energy: 0,
      f0: 220,
      clarity: 0.7,
      fluency: 0.7,
      volume: 0.5,
      pitchStability: 0.8
    },
    timestamp: Date.now()
  });

  const [isConnected, setIsConnected] = useState(false);

  // Connect to voice pipeline (WebSocket or callback)
  useEffect(() => {
    // TODO: Replace with actual voice pipeline connection
    // Example: WebSocket connection
    const connectToVoicePipeline = () => {
      // Mock connection for now
      setIsConnected(true);
      
      // Simulate voice data updates
      const interval = setInterval(() => {
        const newData: VoiceData = {
          metrics: {
            energy: Math.random(),
            f0: 200 + Math.random() * 100,
            clarity: 0.6 + Math.random() * 0.3,
            fluency: 0.6 + Math.random() * 0.3,
            volume: Math.random(),
            pitchStability: 0.7 + Math.random() * 0.2
          },
          waveform: new Float32Array(512).map(() => Math.random() * 2 - 1),
          spectrum: new Float32Array(128).map(() => Math.random()),
          timestamp: Date.now()
        };

        // Smooth updates
        setVoiceData(prev => ({
          ...prev,
          metrics: {
            energy: prev.metrics.energy * (1 - smoothing) + newData.metrics.energy * smoothing,
            f0: prev.metrics.f0 * (1 - smoothing) + newData.metrics.f0 * smoothing,
            clarity: prev.metrics.clarity * (1 - smoothing) + newData.metrics.clarity * smoothing,
            fluency: prev.metrics.fluency * (1 - smoothing) + newData.metrics.fluency * smoothing,
            volume: prev.metrics.volume * (1 - smoothing) + newData.metrics.volume * smoothing,
            pitchStability: prev.metrics.pitchStability * (1 - smoothing) + newData.metrics.pitchStability * smoothing
          },
          waveform: newData.waveform,
          spectrum: newData.spectrum,
          timestamp: newData.timestamp
        }));

        onUpdate?.(newData);
      }, updateInterval);

      return () => clearInterval(interval);
    };

    const cleanup = connectToVoicePipeline();
    return cleanup;
  }, [updateInterval, smoothing, onUpdate]);

  const connect = useCallback(() => {
    setIsConnected(true);
  }, []);

  const disconnect = useCallback(() => {
    setIsConnected(false);
  }, []);

  return {
    voiceData,
    isConnected,
    connect,
    disconnect
  };
};

