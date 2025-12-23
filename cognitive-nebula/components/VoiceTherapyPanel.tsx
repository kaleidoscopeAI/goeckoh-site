import React, { useState, useEffect } from 'react';
import type { Metrics } from '../types';

interface VoiceMetrics {
  energy: number;
  f0: number;
  clarity: number;
  fluency: number;
  volume: number;
  pitchStability: number;
}

interface VoiceTherapyPanelProps {
  voiceMetrics?: VoiceMetrics;
  sessionTime: number;
  exercisesCompleted: number;
  onStartExercise?: () => void;
  onPause?: () => void;
  isActive: boolean;
}

/**
 * Therapeutic voice feedback panel
 * Provides real-time feedback and progress tracking for speech therapy
 */
const VoiceTherapyPanel: React.FC<VoiceTherapyPanelProps> = ({
  voiceMetrics,
  sessionTime,
  exercisesCompleted,
  onStartExercise,
  onPause,
  isActive
}) => {
  const [sessionStartTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - sessionStartTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isActive, sessionStartTime]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getClarityColor = (clarity: number) => {
    if (clarity > 0.7) return 'text-green-400';
    if (clarity > 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getFluencyColor = (fluency: number) => {
    if (fluency > 0.7) return 'text-green-400';
    if (fluency > 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-lg rounded-xl p-6 border border-cyan-500/30 shadow-2xl max-w-md w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-cyan-300 font-bold text-lg">Voice Therapy</h3>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-xs text-gray-400">{isActive ? 'Active' : 'Paused'}</span>
        </div>
      </div>

      {/* Session Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Session Time</div>
          <div className="text-xl font-mono text-cyan-300">{formatTime(elapsedTime)}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Exercises</div>
          <div className="text-xl font-mono text-cyan-300">{exercisesCompleted}</div>
        </div>
      </div>

      {/* Voice Metrics */}
      {voiceMetrics && (
        <div className="space-y-3 mb-4">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Clarity</span>
              <span className={getClarityColor(voiceMetrics.clarity)}>
                {(voiceMetrics.clarity * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  voiceMetrics.clarity > 0.7 ? 'bg-green-400' :
                  voiceMetrics.clarity > 0.4 ? 'bg-yellow-400' : 'bg-red-400'
                }`}
                style={{ width: `${voiceMetrics.clarity * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Fluency</span>
              <span className={getFluencyColor(voiceMetrics.fluency)}>
                {(voiceMetrics.fluency * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  voiceMetrics.fluency > 0.7 ? 'bg-green-400' :
                  voiceMetrics.fluency > 0.4 ? 'bg-yellow-400' : 'bg-red-400'
                }`}
                style={{ width: `${voiceMetrics.fluency * 100}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-gray-900/50 rounded p-2">
              <div className="text-gray-400">Energy</div>
              <div className="text-cyan-300 font-mono">{(voiceMetrics.energy * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-gray-900/50 rounded p-2">
              <div className="text-gray-400">Pitch</div>
              <div className="text-cyan-300 font-mono">{voiceMetrics.f0.toFixed(0)} Hz</div>
            </div>
            <div className="bg-gray-900/50 rounded p-2">
              <div className="text-gray-400">Volume</div>
              <div className="text-cyan-300 font-mono">{(voiceMetrics.volume * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-2">
        {!isActive ? (
          <button
            onClick={onStartExercise}
            className="flex-1 bg-cyan-600 hover:bg-cyan-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Start Exercise
          </button>
        ) : (
          <button
            onClick={onPause}
            className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Pause
          </button>
        )}
      </div>
    </div>
  );
};

export default VoiceTherapyPanel;

