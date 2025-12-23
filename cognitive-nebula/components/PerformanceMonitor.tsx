import React, { useEffect, useState, useRef } from 'react';

interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  memoryUsage?: number;
  renderTime: number;
}

interface PerformanceMonitorProps {
  enabled?: boolean;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  showDetails?: boolean;
}

/**
 * Performance monitoring component
 * Displays FPS, frame time, and other performance metrics
 */
const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  enabled = false,
  position = 'top-right',
  showDetails = false
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    frameTime: 16.67,
    renderTime: 0
  });
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    if (!enabled) return;

    const measure = () => {
      const now = performance.now();
      const delta = now - lastTimeRef.current;
      frameCountRef.current++;

      if (delta >= 1000) {
        const fps = Math.round((frameCountRef.current * 1000) / delta);
        const frameTime = delta / frameCountRef.current;

        setMetrics(prev => ({
          ...prev,
          fps,
          frameTime: Math.round(frameTime * 100) / 100
        }));

        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }

      animationFrameRef.current = requestAnimationFrame(measure);
    };

    animationFrameRef.current = requestAnimationFrame(measure);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled]);

  if (!enabled) return null;

  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4'
  };

  const fpsColor = metrics.fps >= 55 ? 'text-green-400' :
                   metrics.fps >= 30 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div className={`fixed ${positionClasses[position]} bg-black/80 backdrop-blur-lg rounded-lg p-3 border border-cyan-500/30 text-white font-mono text-xs z-50`}>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-gray-400">FPS:</span>
        <span className={fpsColor}>{metrics.fps}</span>
      </div>
      {showDetails && (
        <>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-gray-400">Frame:</span>
            <span className="text-cyan-400">{metrics.frameTime}ms</span>
          </div>
          {metrics.memoryUsage && (
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Memory:</span>
              <span className="text-cyan-400">{(metrics.memoryUsage / 1024 / 1024).toFixed(1)}MB</span>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default PerformanceMonitor;

