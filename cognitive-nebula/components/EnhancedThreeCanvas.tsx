import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import type { Metrics, Settings } from '../types';

interface EnhancedThreeCanvasProps {
  metrics: Metrics;
  aiThought: string;
  imageData: string | null;
  settings: Settings;
  voiceData?: {
    bubbleState?: {
      radius: number;
      color_r: number;
      color_g: number;
      color_b: number;
      rough: number;
      metal: number;
      spike: number;
      energy?: number;
      f0?: number;
    };
    waveform?: Float32Array;
    spectrum?: Float32Array;
  };
}

const MAX_NODES = 30000;

// Enhanced visual embedding with voice influence
const createEnhancedEmbedding = (
  thought: string,
  numNodes: number,
  saturationMultiplier: number,
  voiceEnergy: number = 0
): { positions: THREE.Vector3[], colors: THREE.Color[] } => {
  const positions: THREE.Vector3[] = [];
  const colors: THREE.Color[] = [];
  const hash = Array.from(thought).reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const baseHue = (hash % 360) / 360;
  
  const phi = Math.PI * (3 - Math.sqrt(5));
  const baseRadius = 350 + voiceEnergy * 100; // Voice energy expands the nebula
  const noiseFactor = 100 + voiceEnergy * 50;
  const hashSeed1 = Math.sin(hash * 0.01) * 5;
  const hashSeed2 = Math.cos(hash * 0.01) * 5;

  for (let i = 0; i < numNodes; i++) {
    const y = 1 - (i / (numNodes - 1)) * 2;
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = phi * i;

    const x = Math.cos(theta) * radiusAtY;
    const z = Math.sin(theta) * radiusAtY;
    
    const noise = (Math.sin(x * hashSeed1) + Math.cos(y * hashSeed2)) * noiseFactor;
    const finalRadius = baseRadius + noise;
    
    positions.push(new THREE.Vector3(x * finalRadius, y * finalRadius, z * finalRadius));

    // Enhanced color with voice energy influence
    const hue = (baseHue + (Math.sin(i * 0.05 + hashSeed1) * 0.1)) % 1.0;
    const saturation = (0.6 + Math.sin(i * 0.02) * 0.4 + voiceEnergy * 0.3) * saturationMultiplier;
    const lightness = 0.5 + Math.sin(i * 0.03) * 0.25 + voiceEnergy * 0.2;
    colors.push(new THREE.Color().setHSL(hue, saturation, lightness));
  }
  return { positions, colors };
};

const EnhancedThreeCanvas: React.FC<EnhancedThreeCanvasProps> = ({
  metrics,
  aiThought,
  imageData,
  settings,
  voiceData
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  
  const voiceEnergy = voiceData?.bubbleState?.energy || 0;
  const initialNodeCount = Math.floor(MAX_NODES * settings.particleCount);
  const initialEmbedding = createEnhancedEmbedding(
    "A silent nebula awaits a spark of inquiry.",
    initialNodeCount,
    settings.colorSaturation,
    voiceEnergy
  );
  const targetPositionsRef = useRef<THREE.Vector3[]>(initialEmbedding.positions);
  const targetColorsRef = useRef<THREE.Color[]>(initialEmbedding.colors);
  const activeNodesRef = useRef<number>(initialNodeCount);

  const metricsRef = useRef(metrics);
  useEffect(() => { metricsRef.current = metrics; }, [metrics]);

  const imageDataRef = useRef(imageData);
  useEffect(() => { imageDataRef.current = imageData; }, [imageData]);
  
  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  const voiceDataRef = useRef(voiceData);
  useEffect(() => { voiceDataRef.current = voiceData; }, [voiceData]);

  useEffect(() => {
    const numNodes = Math.floor(MAX_NODES * settings.particleCount);
    const currentVoiceEnergy = voiceDataRef.current?.bubbleState?.energy || 0;
    
    if (imageData) {
      const img = new Image();
      img.onload = () => {
        // Enhanced image embedding with voice influence
        const canvas = document.createElement('canvas');
        const downscaleFactor = Math.max(1, Math.floor(img.width / 256));
        const width = img.width / downscaleFactor;
        const height = img.height / downscaleFactor;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        ctx.drawImage(img, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);
        const pixels = imageData.data;

        const newPositions: THREE.Vector3[] = [];
        const newColors: THREE.Color[] = [];
        const scale = 500 + currentVoiceEnergy * 200;
        const depth = 200 + currentVoiceEnergy * 100;

        const step = 4;
        const jitter = step / 2;

        for (let y = 0; y < height; y += step) {
          for (let x = 0; x < width; x += step) {
            if (newPositions.length >= numNodes) break;

            const jitterX = x + Math.floor(Math.random() * jitter * 2) - jitter;
            const jitterY = y + Math.floor(Math.random() * jitter * 2) - jitter;
            
            const finalX = Math.max(0, Math.min(width - 1, jitterX));
            const finalY = Math.max(0, Math.min(height - 1, jitterY));
            
            const i = (finalY * width + finalX) * 4;
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            const a = pixels[i + 3];

            if (a > 50) {
              const brightness = (r + g + b) / (3 * 255);
              const posX = (finalX / width - 0.5) * scale;
              const posY = -(finalY / height - 0.5) * scale;
              const posZ = (brightness - 0.5) * depth;

              newPositions.push(new THREE.Vector3(posX, posY, posZ));
              // Enhance colors with voice energy
              const color = new THREE.Color(r / 255, g / 255, b / 255);
              color.lerp(new THREE.Color(1, 0.5, 0), currentVoiceEnergy * 0.2);
              newColors.push(color);
            }
          }
          if (newPositions.length >= numNodes) break;
        }
        
        if (newPositions.length > 0) {
          targetPositionsRef.current = newPositions;
          targetColorsRef.current = newColors;
          activeNodesRef.current = newPositions.length;
        }
      };
      img.onerror = () => {
        const { positions, colors } = createEnhancedEmbedding(
          aiThought,
          numNodes,
          settings.colorSaturation,
          currentVoiceEnergy
        );
        targetPositionsRef.current = positions;
        targetColorsRef.current = colors;
        activeNodesRef.current = numNodes;
      };
      img.src = `data:image/png;base64,${imageData}`;
    } else {
      const { positions, colors } = createEnhancedEmbedding(
        aiThought,
        numNodes,
        settings.colorSaturation,
        currentVoiceEnergy
      );
      targetPositionsRef.current = positions;
      targetColorsRef.current = colors;
      activeNodesRef.current = numNodes;
    }
  }, [imageData, aiThought, settings.particleCount, settings.colorSaturation, voiceEnergy]);

  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.FogExp2(0x000000, 0.001);
    
    const camera = new THREE.PerspectiveCamera(
      75,
      mount.clientWidth / mount.clientHeight,
      0.1,
      2000
    );
    camera.position.z = 600;
    
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.autoClearColor = false;
    mount.appendChild(renderer.domElement);
    
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 100;
    controls.maxDistance = 1200;

    // Enhanced background stars with voice-reactive twinkling
    const starGeometry = new THREE.BufferGeometry();
    const starVertices = [];
    const starColors = [];
    for (let i = 0; i < 20000; i++) {
      const x = (Math.random() - 0.5) * 2500;
      const y = (Math.random() - 0.5) * 2500;
      const z = (Math.random() - 0.5) * 2500;
      starVertices.push(x, y, z);
      starColors.push(1, 1, 1);
    }
    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
    starGeometry.setAttribute('color', new THREE.Float32BufferAttribute(starColors, 3));
    const starMaterial = new THREE.PointsMaterial({
      vertexColors: true,
      size: 0.8,
      transparent: true,
      opacity: 0.7
    });
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // Fade effect for trails
    const fadeScene = new THREE.Scene();
    const fadeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fadeMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
    });
    const fadePlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fadeMaterial);
    fadeScene.add(fadePlane);
    
    const geometry = new THREE.SphereGeometry(1.5, 12, 12); // Higher resolution
    const material = new THREE.MeshStandardMaterial({
      emissive: new THREE.Color(0x222222),
      roughness: 0.5,
      metalness: 0.3
    });
    const group = new THREE.Group();
    scene.add(group);
    
    const nodes: THREE.Mesh[] = [];
    for (let i = 0; i < MAX_NODES; i++) {
      const nodeMaterial = material.clone();
      nodeMaterial.color.setHSL(Math.random(), 0.7, 0.5);
      const node = new THREE.Mesh(geometry, nodeMaterial);
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 500 + Math.random() * 200;
      node.position.set(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      );
      node.scale.set(0, 0, 0);
      nodes.push(node);
      group.add(node);
    }
    
    // Enhanced lighting with voice-reactive colors
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0xffffff, 1, 1000);
    pointLight1.position.set(200, 200, 200);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x00ffff, 0.8, 1000);
    pointLight2.position.set(-200, -200, -200);
    scene.add(pointLight2);
    
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(Infinity, Infinity);
    const attractorRef = React.createRef<THREE.Vector3 | null>();
    attractorRef.current = null;

    // Post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(mount.clientWidth, mount.clientHeight),
      1.2,
      0.4,
      0.85
    );
    composer.addPass(bloomPass);

    const filmPass = new FilmPass({
      noiseIntensity: 0.05,
      scanlinesIntensity: 0.02,
      scanlinesCount: 1024,
      grayscale: false
    });
    composer.addPass(filmPass);

    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.01;

      const currentSettings = settingsRef.current;
      const currentMetrics = metricsRef.current;
      const currentVoiceData = voiceDataRef.current;
      const voiceEnergy = currentVoiceData?.bubbleState?.energy || 0;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(nodes);

      if (intersects.length > 0) {
        attractorRef.current = intersects[0].object.position;
      } else {
        attractorRef.current = null;
      }
      
      // Enhanced rotation with voice influence
      const rotationSpeed = (0.0001 + voiceEnergy * 0.0002) * currentMetrics.arousal * currentSettings.movementSpeed;
      group.rotation.x += rotationSpeed;
      group.rotation.y += rotationSpeed * 2;

      // Voice-reactive star twinkling
      stars.visible = currentSettings.showStars;
      if (stars.visible) {
        stars.rotation.y += 0.00005;
        const starColors = starGeometry.attributes.color.array as Float32Array;
        for (let i = 0; i < starColors.length; i += 3) {
          const twinkle = Math.sin(time * 2 + i) * 0.3 + 0.7;
          starColors[i] = twinkle;
          starColors[i + 1] = twinkle;
          starColors[i + 2] = twinkle;
        }
        starGeometry.attributes.color.needsUpdate = true;
      }

      const isImageState = !!imageDataRef.current;
      const positionLerp = isImageState ? 0.02 : 0.01;
      const colorLerp = 0.04;
      const scaleLerp = 0.05;
      const targetPositions = targetPositionsRef.current;
      const targetColors = targetColorsRef.current;
      const activeNodes = activeNodesRef.current;
      const attractor = attractorRef.current;

      nodes.forEach((node, i) => {
        const isActive = i < activeNodes;
        const targetScale = isActive ? (1.0 + voiceEnergy * 0.3) : 0.0;
        node.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), scaleLerp);

        if (isActive) {
          const baseTargetPos = targetPositions[i];
          if (!baseTargetPos) return;

          const fourD_amplitude = 5 + currentMetrics.confusion * 30 + voiceEnergy * 20;
          const fourD_speed = (0.3 + currentMetrics.arousal * 1.0 + voiceEnergy * 0.5) * currentSettings.movementSpeed;

          const time_i = time + i * 0.31;
          const offsetVector = new THREE.Vector3(
            (Math.sin(time_i * fourD_speed) + Math.cos(time_i * fourD_speed * 0.41)) * 0.7,
            (Math.cos(time_i * fourD_speed * 0.63) + Math.sin(time_i * fourD_speed * 0.83)) * 0.7,
            (Math.sin(time_i * fourD_speed * 1.13) + Math.cos(time_i * fourD_speed * 0.57)) * 0.7
          );
          offsetVector.multiplyScalar(fourD_amplitude);
          
          const dynamicTargetPos = new THREE.Vector3().copy(baseTargetPos).add(offsetVector);
          node.position.lerp(dynamicTargetPos, positionLerp * currentMetrics.coherence);

          if (attractor) {
            const attractionRadius = 150;
            const distance = node.position.distanceTo(attractor);
            if (distance < attractionRadius) {
              const attractionStrength = (1 - distance / attractionRadius) * 0.015 * currentMetrics.resonance;
              node.position.lerp(attractor, attractionStrength);
            }
          }
          
          const targetColor = targetColors[i];
          if (targetColor) {
            // Voice energy enhances color intensity
            const enhancedColor = targetColor.clone();
            enhancedColor.lerp(new THREE.Color(1, 1, 1), voiceEnergy * 0.2);
            node.material.color.lerp(enhancedColor, colorLerp);
          }
        }
      });

      // Voice-reactive lighting
      if (currentVoiceData?.bubbleState) {
        const bubbleColor = new THREE.Color(
          currentVoiceData.bubbleState.color_r,
          currentVoiceData.bubbleState.color_g,
          currentVoiceData.bubbleState.color_b
        );
        pointLight2.color.lerp(bubbleColor, 0.1);
        pointLight2.intensity = 0.8 + voiceEnergy * 0.5;
      }

      controls.update();
      
      const fadeOpacity = currentSettings.showTrails
        ? (0.4 - currentMetrics.arousal * 0.35 - voiceEnergy * 0.1)
        : 1.0;
      fadeMaterial.opacity = Math.max(0.05, Math.min(1.0, fadeOpacity));
      
      composer.render();
    };
    animate();

    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      composer.setSize(mount.clientWidth, mount.clientHeight);
    };

    const handleMouseMove = (event: MouseEvent) => {
      const rect = mount.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      cancelAnimationFrame(animate as any);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      if (mount && renderer.domElement) {
        mount.removeChild(renderer.domElement);
      }
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      nodes.forEach(node => node.material.dispose());
      starGeometry.dispose();
      starMaterial.dispose();
    };
  }, []);

  return <div ref={mountRef} className="absolute inset-0 w-full h-full" />;
};

export default EnhancedThreeCanvas;

