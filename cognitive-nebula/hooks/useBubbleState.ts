import { useState, useEffect, useMemo } from 'react';

interface BubbleState {
  radius: number;
  color_r: number;
  color_g: number;
  color_b: number;
  rough: number;
  metal: number;
  spike: number;
  energy?: number;
  f0?: number;
}

interface VoiceMetrics {
  energy: number;
  f0: number;
  zcr: number;
  tilt: number;
  hnr: number;
}

interface UseBubbleStateOptions {
  baseRadius?: number;
  baseColor?: { r: number; g: number; b: number };
  smoothing?: number;
}

/**
 * Custom hook for computing bubble state from voice metrics
 * Maps voice data to visual bubble properties
 */
export const useBubbleState = (
  voiceMetrics: VoiceMetrics | null,
  options: UseBubbleStateOptions = {}
) => {
  const {
    baseRadius = 1.0,
    baseColor = { r: 0.25, g: 0.7, b: 1.0 },
    smoothing = 0.1
  } = options;

  const [bubbleState, setBubbleState] = useState<BubbleState>({
    radius: baseRadius,
    color_r: baseColor.r,
    color_g: baseColor.g,
    color_b: baseColor.b,
    rough: 0.4,
    metal: 0.6,
    spike: 0.0,
    energy: 0,
    f0: 220
  });

  useEffect(() => {
    if (!voiceMetrics) return;

    // Compute bubble properties from voice metrics
    const newState: BubbleState = {
      // Radius: base + energy expansion
      radius: baseRadius * (1.0 + voiceMetrics.energy * 0.5),
      
      // Color: F0-based hue (80-400Hz mapped to 0-1)
      color_r: baseColor.r,
      color_g: baseColor.g,
      color_b: baseColor.b,
      
      // Roughness: inverse of HNR (higher HNR = smoother)
      rough: Math.max(0, Math.min(1, 1.0 - voiceMetrics.hnr)),
      
      // Metalness: based on spectral tilt
      metal: Math.max(0, Math.min(1, 0.5 + voiceMetrics.tilt / 5.0)),
      
      // Spike: Bouba/Kiki effect from ZCR
      spike: Math.max(0, Math.min(1, voiceMetrics.zcr * 2.0)),
      
      // Raw metrics
      energy: voiceMetrics.energy,
      f0: voiceMetrics.f0
    };

    // Smooth transitions
    setBubbleState(prev => ({
      radius: prev.radius * (1 - smoothing) + newState.radius * smoothing,
      color_r: prev.color_r * (1 - smoothing) + newState.color_r * smoothing,
      color_g: prev.color_g * (1 - smoothing) + newState.color_g * smoothing,
      color_b: prev.color_b * (1 - smoothing) + newState.color_b * smoothing,
      rough: prev.rough * (1 - smoothing) + newState.rough * smoothing,
      metal: prev.metal * (1 - smoothing) + newState.metal * smoothing,
      spike: prev.spike * (1 - smoothing) + newState.spike * smoothing,
      energy: newState.energy,
      f0: newState.f0
    }));
  }, [voiceMetrics, baseRadius, baseColor, smoothing]);

  // Compute color from F0
  const colorFromF0 = useMemo(() => {
    if (!voiceMetrics) return baseColor;
    
    const f0Norm = Math.max(0, Math.min(1, (voiceMetrics.f0 - 80) / 320));
    const hue = f0Norm * 0.7; // Blue to cyan range
    
    // Convert HSL to RGB (simplified)
    const c = 0.7;
    const x = c * (1 - Math.abs((hue * 6) % 2 - 1));
    const m = 0.3;
    
    let r = 0, g = 0, b = 0;
    if (hue < 1/6) { r = c; g = x; b = 0; }
    else if (hue < 2/6) { r = x; g = c; b = 0; }
    else if (hue < 3/6) { r = 0; g = c; b = x; }
    else if (hue < 4/6) { r = 0; g = x; b = c; }
    else if (hue < 5/6) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    
    return {
      r: r + m,
      g: g + m,
      b: b + m
    };
  }, [voiceMetrics, baseColor]);

  return {
    bubbleState: {
      ...bubbleState,
      color_r: colorFromF0.r,
      color_g: colorFromF0.g,
      color_b: colorFromF0.b
    }
  };
};

