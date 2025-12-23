import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface VoiceFeedbackData {
  waveform: Float32Array;
  spectrum: Float32Array;
  f0: number;
  energy: number;
  clarity: number;
  volume: number;
}

interface RealTimeVoiceFeedbackProps {
  voiceData: VoiceFeedbackData;
  size?: 'small' | 'medium' | 'large';
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

/**
 * Real-time voice feedback visualization
 * Shows waveform, spectrum, and voice metrics in a compact overlay
 */
const RealTimeVoiceFeedback: React.FC<RealTimeVoiceFeedbackProps> = ({
  voiceData,
  size = 'medium',
  position = 'bottom-right'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  const sizeMap = {
    small: { width: 200, height: 100 },
    medium: { width: 300, height: 150 },
    large: { width: 400, height: 200 }
  };

  const positionMap = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4'
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = sizeMap[size];
    canvas.width = width;
    canvas.height = height;

    const draw = () => {
      // Clear canvas
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(0, 0, width, height);

      // Draw waveform
      if (voiceData.waveform && voiceData.waveform.length > 0) {
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const samples = Math.min(voiceData.waveform.length, width);
        const step = width / samples;
        
        for (let i = 0; i < samples; i++) {
          const x = i * step;
          const y = (height / 2) + (voiceData.waveform[i] * height * 0.4);
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      // Draw spectrum bars
      if (voiceData.spectrum && voiceData.spectrum.length > 0) {
        const barCount = Math.min(voiceData.spectrum.length, 32);
        const barWidth = width / barCount;
        
        for (let i = 0; i < barCount; i++) {
          const magnitude = voiceData.spectrum[i];
          const barHeight = magnitude * height * 0.6;
          
          // Color based on frequency
          const hue = (i / barCount) * 0.7; // Blue to cyan
          ctx.fillStyle = `hsl(${hue * 360}, 70%, 60%)`;
          ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
        }
      }

      // Draw F0 indicator
      ctx.fillStyle = '#a3e635';
      ctx.font = '12px monospace';
      ctx.fillText(`F0: ${voiceData.f0.toFixed(0)} Hz`, 5, 15);
      
      // Draw energy indicator
      ctx.fillStyle = '#22d3ee';
      ctx.fillText(`Energy: ${(voiceData.energy * 100).toFixed(0)}%`, 5, 30);
      
      // Draw clarity indicator
      const clarityColor = voiceData.clarity > 0.7 ? '#22c55e' : 
                           voiceData.clarity > 0.4 ? '#eab308' : '#ef4444';
      ctx.fillStyle = clarityColor;
      ctx.fillText(`Clarity: ${(voiceData.clarity * 100).toFixed(0)}%`, 5, 45);

      animationFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [voiceData, size]);

  return (
    <div className={`absolute ${positionMap[position]} bg-black/70 backdrop-blur-lg rounded-lg p-2 border border-cyan-500/30`}>
      <canvas
        ref={canvasRef}
        className="block"
        style={{ width: sizeMap[size].width, height: sizeMap[size].height }}
      />
    </div>
  );
};

export default RealTimeVoiceFeedback;

