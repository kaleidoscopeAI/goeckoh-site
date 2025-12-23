import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

interface DataPoint {
  timestamp: number;
  value: number;
  label?: string;
}

interface AdvancedDataVisualizationProps {
  data: DataPoint[];
  type: 'line' | 'bar' | 'area' | '3d' | 'heatmap';
  title?: string;
  color?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
}

/**
 * Advanced data visualization component
 * Supports multiple chart types with 3D capabilities
 */
const AdvancedDataVisualization: React.FC<AdvancedDataVisualizationProps> = ({
  data,
  type,
  title,
  color = '#22d3ee',
  height = 200,
  showGrid = true,
  showLegend = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (type === '3d') {
      render3D();
    } else {
      render2D();
    }
  }, [data, type, color]);

  const render2D = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const chartHeight = height - 40;
    const padding = 20;

    // Clear
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(0, 0, width, height);

    if (data.length === 0) return;

    // Find min/max
    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
      }
    }

    // Draw chart
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;

    if (type === 'line' || type === 'area') {
      ctx.beginPath();
      const step = (width - padding * 2) / (data.length - 1 || 1);
      
      data.forEach((point, i) => {
        const x = padding + i * step;
        const normalizedValue = (point.value - min) / range;
        const y = padding + chartHeight - (normalizedValue * chartHeight);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      if (type === 'area') {
        ctx.lineTo(width - padding, padding + chartHeight);
        ctx.lineTo(padding, padding + chartHeight);
        ctx.closePath();
        ctx.fillStyle = color + '40';
        ctx.fill();
      }
      
      ctx.stroke();
    } else if (type === 'bar') {
      const barWidth = (width - padding * 2) / data.length;
      data.forEach((point, i) => {
        const x = padding + i * barWidth;
        const normalizedValue = (point.value - min) / range;
        const barHeight = normalizedValue * chartHeight;
        const y = padding + chartHeight - barHeight;
        
        ctx.fillRect(x, y, barWidth - 2, barHeight);
      });
    } else if (type === 'heatmap') {
      const cellWidth = (width - padding * 2) / data.length;
      data.forEach((point, i) => {
        const normalizedValue = (point.value - min) / range;
        const intensity = Math.floor(normalizedValue * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`;
        ctx.fillRect(padding + i * cellWidth, padding, cellWidth - 2, chartHeight);
      });
    }

    // Draw title
    if (title) {
      ctx.fillStyle = '#e5e7eb';
      ctx.font = '14px sans-serif';
      ctx.fillText(title, padding, 15);
    }
  };

  const render3D = () => {
    const container = containerRef.current;
    if (!container) return;

    // Clear previous
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, height);
    container.appendChild(renderer.domElement);

    // Create 3D line
    const points: THREE.Vector3[] = [];
    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const range = maxValue - minValue || 1;

    data.forEach((point, i) => {
      const x = (i / data.length) * 10 - 5;
      const y = ((point.value - minValue) / range) * 5;
      const z = Math.sin(i * 0.1) * 2;
      points.push(new THREE.Vector3(x, y, z));
    });

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: new THREE.Color(color) });
    const line = new THREE.Line(geometry, material);
    scene.add(line);

    camera.position.set(0, 2, 8);
    camera.lookAt(0, 0, 0);

    const animate = () => {
      requestAnimationFrame(animate);
      line.rotation.y += 0.01;
      renderer.render(scene, camera);
    };
    animate();
  };

  if (type === '3d') {
    return (
      <div ref={containerRef} className="w-full" style={{ height: `${height}px` }} />
    );
  }

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        className="w-full h-auto"
      />
    </div>
  );
};

export default AdvancedDataVisualization;

