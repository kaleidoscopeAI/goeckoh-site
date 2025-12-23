import React from 'react';

interface LoadingStatesProps {
  type?: 'spinner' | 'pulse' | 'skeleton' | 'progress';
  message?: string;
  progress?: number;
  size?: 'small' | 'medium' | 'large';
}

/**
 * Enhanced loading states with multiple styles
 * Provides visual feedback during async operations
 */
const LoadingStates: React.FC<LoadingStatesProps> = ({
  type = 'spinner',
  message,
  progress,
  size = 'medium'
}) => {
  const sizeClasses = {
    small: 'w-4 h-4',
    medium: 'w-8 h-8',
    large: 'w-12 h-12'
  };

  if (type === 'spinner') {
    return (
      <div className="flex flex-col items-center justify-center gap-2">
        <div className={`${sizeClasses[size]} border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin`} />
        {message && <p className="text-sm text-gray-400">{message}</p>}
      </div>
    );
  }

  if (type === 'pulse') {
    return (
      <div className="flex flex-col items-center justify-center gap-2">
        <div className={`${sizeClasses[size]} bg-cyan-400 rounded-full animate-pulse`} />
        {message && <p className="text-sm text-gray-400">{message}</p>}
      </div>
    );
  }

  if (type === 'skeleton') {
    return (
      <div className="space-y-2 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-3/4" />
        <div className="h-4 bg-gray-700 rounded w-1/2" />
        <div className="h-4 bg-gray-700 rounded w-5/6" />
      </div>
    );
  }

  if (type === 'progress') {
    return (
      <div className="flex flex-col items-center justify-center gap-2 w-full">
        {message && <p className="text-sm text-gray-400">{message}</p>}
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div
            className="bg-cyan-400 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress || 0}%` }}
          />
        </div>
        {progress !== undefined && (
          <p className="text-xs text-gray-500">{progress.toFixed(0)}%</p>
        )}
      </div>
    );
  }

  return null;
};

export default LoadingStates;

