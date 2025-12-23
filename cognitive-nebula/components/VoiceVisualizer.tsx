import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

interface VoiceVisualizerProps {
  waveform?: Float32Array;
  spectrum?: Float32Array;
  f0?: number;
  energy?: number;
  color?: THREE.Color;
}

/**
 * Advanced voice visualization component
 * Displays real-time waveform, frequency spectrum, and pitch visualization
 */
const VoiceVisualizer: React.FC<VoiceVisualizerProps> = ({
  waveform,
  spectrum,
  f0 = 220,
  energy = 0,
  color = new THREE.Color(0.25, 0.7, 1.0)
}) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(
      75,
      mount.clientWidth / mount.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 10);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    // Waveform visualization
    const waveformGroup = new THREE.Group();
    scene.add(waveformGroup);

    // Frequency spectrum bars
    const spectrumGroup = new THREE.Group();
    scene.add(spectrumGroup);

    // Pitch indicator (circular visualization)
    const pitchGroup = new THREE.Group();
    scene.add(pitchGroup);

    // Create waveform line
    const waveformGeometry = new THREE.BufferGeometry();
    const waveformMaterial = new THREE.LineBasicMaterial({
      color: color,
      linewidth: 2
    });
    const waveformLine = new THREE.Line(waveformGeometry, waveformMaterial);
    waveformGroup.add(waveformLine);

    // Create spectrum bars
    const barCount = 128;
    const bars: THREE.Mesh[] = [];
    for (let i = 0; i < barCount; i++) {
      const barGeometry = new THREE.BoxGeometry(0.05, 0.1, 0.05);
      const barMaterial = new THREE.MeshBasicMaterial({ color: color });
      const bar = new THREE.Mesh(barGeometry, barMaterial);
      bar.position.x = (i - barCount / 2) * 0.1;
      bar.position.y = -3;
      spectrumGroup.add(bar);
      bars.push(bar);
    }

    // Pitch visualization - circular frequency display
    const pitchGeometry = new THREE.RingGeometry(2, 2.5, 64);
    const pitchMaterial = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide
    });
    const pitchRing = new THREE.Mesh(pitchGeometry, pitchMaterial);
    pitchRing.position.y = 2;
    pitchGroup.add(pitchRing);

    // Energy-based particles
    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 1000;
    const particlePositions = new Float32Array(particleCount * 3);
    const particleColors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      particlePositions[i3] = (Math.random() - 0.5) * 10;
      particlePositions[i3 + 1] = (Math.random() - 0.5) * 10;
      particlePositions[i3 + 2] = (Math.random() - 0.5) * 10;
      
      particleColors[i3] = color.r;
      particleColors[i3 + 1] = color.g;
      particleColors[i3 + 2] = color.b;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));
    
    const particleMaterial = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(color, 1, 100);
    pointLight.position.set(0, 0, 5);
    scene.add(pointLight);

    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.016;

      // Update waveform
      if (waveform && waveform.length > 0) {
        const points: THREE.Vector3[] = [];
        const sampleCount = Math.min(waveform.length, 512);
        const scale = 3;
        
        for (let i = 0; i < sampleCount; i++) {
          const x = (i / sampleCount - 0.5) * 8;
          const y = waveform[i] * scale * (1 + energy);
          points.push(new THREE.Vector3(x, y, 0));
        }
        
        waveformGeometry.setFromPoints(points);
        waveformGeometry.attributes.position.needsUpdate = true;
      }

      // Update spectrum bars
      if (spectrum && spectrum.length > 0) {
        bars.forEach((bar, i) => {
          const spectrumIndex = Math.floor((i / barCount) * spectrum.length);
          const magnitude = spectrum[spectrumIndex] || 0;
          const height = Math.max(0.1, magnitude * 5 * (1 + energy));
          
          bar.scale.y = height;
          bar.position.y = -3 + height / 2;
          
          // Color based on frequency
          const freq = (i / barCount) * 20000; // 0-20kHz
          const hue = (freq / 20000) * 0.7; // Blue to cyan
          bar.material.color.setHSL(hue, 0.8, 0.6);
        });
      }

      // Update pitch ring
      const f0Norm = (f0 - 80) / 320; // Normalize 80-400Hz
      pitchRing.scale.setScalar(0.8 + f0Norm * 0.4);
      pitchRing.rotation.z += 0.01;
      
      // Pulse with energy
      const pulse = 1 + Math.sin(time * 5) * energy * 0.2;
      pitchRing.material.opacity = 0.4 + energy * 0.4;
      pitchRing.scale.multiplyScalar(pulse);

      // Animate particles
      const positions = particleGeometry.attributes.position.array as Float32Array;
      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        positions[i3 + 1] += Math.sin(time + i) * 0.01 * energy;
        if (positions[i3 + 1] > 5) positions[i3 + 1] = -5;
      }
      particleGeometry.attributes.position.needsUpdate = true;

      // Rotate groups
      waveformGroup.rotation.z += 0.001;
      spectrumGroup.rotation.y += 0.002;

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mount && renderer.domElement) {
        mount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [waveform, spectrum, f0, energy, color]);

  return <div ref={mountRef} className="absolute inset-0 w-full h-full" />;
};

export default VoiceVisualizer;

