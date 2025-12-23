import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import type { Metrics, Settings } from '../types';

interface ThreeCanvasProps {
  metrics: Metrics;
  aiThought: string;
  imageData: string | null;
  settings: Settings;
}

const MAX_NODES = 30000;

// Uses a spherical Fibonacci lattice to create a more organic, even distribution
const createVisualEmbedding = (thought: string, numNodes: number, saturationMultiplier: number): { positions: THREE.Vector3[], colors: THREE.Color[] } => {
    const positions: THREE.Vector3[] = [];
    const colors: THREE.Color[] = [];
    const hash = Array.from(thought).reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const baseHue = (hash % 360) / 360;
    
    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle
    const baseRadius = 350;
    const noiseFactor = 100;
    const hashSeed1 = Math.sin(hash * 0.01) * 5;
    const hashSeed2 = Math.cos(hash * 0.01) * 5;

    const isInitialThought = thought === 'A silent nebula awaits a spark of inquiry.';

    for (let i = 0; i < numNodes; i++) {
        const y = 1 - (i / (numNodes - 1)) * 2; // y goes from 1 to -1
        const radiusAtY = Math.sqrt(1 - y * y);
        const theta = phi * i;

        const x = Math.cos(theta) * radiusAtY;
        const z = Math.sin(theta) * radiusAtY;
        
        const noise = (Math.sin(x * hashSeed1) + Math.cos(y * hashSeed2)) * noiseFactor;
        const finalRadius = baseRadius + noise;
        
        positions.push(new THREE.Vector3(x * finalRadius, y * finalRadius, z * finalRadius));

        const hue = isInitialThought ? Math.random() : (baseHue + (Math.sin(i * 0.05 + hashSeed1) * 0.1)) % 1.0;
        const saturation = (isInitialThought ? 0.5 + Math.random() * 0.3 : 0.6 + Math.sin(i * 0.02) * 0.4) * saturationMultiplier;
        const lightness = isInitialThought ? 0.4 + Math.random() * 0.3 : 0.5 + Math.sin(i * 0.03) * 0.25;
        colors.push(new THREE.Color().setHSL(hue, saturation, lightness));
    }
    return { positions, colors };
}

// Uses jittered grid sampling to break up uniform patterns
const createImageEmbedding = (img: HTMLImageElement, maxNodes: number): { positions: THREE.Vector3[], colors: THREE.Color[] } => {
    const canvas = document.createElement('canvas');
    const downscaleFactor = Math.max(1, Math.floor(img.width / 256));
    const width = img.width / downscaleFactor;
    const height = img.height / downscaleFactor;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return { positions: [], colors: [] };

    ctx.drawImage(img, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    const pixels = imageData.data;

    const newPositions: THREE.Vector3[] = [];
    const newColors: THREE.Color[] = [];
    const scale = 500;
    const depth = 200;

    const step = 4; // Sample from a 4x4 grid cell
    const jitter = step / 2; // Jitter within that cell

    for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
            if (newPositions.length >= maxNodes) break;

            const jitterX = x + Math.floor(Math.random() * jitter * 2) - jitter;
            const jitterY = y + Math.floor(Math.random() * jitter * 2) - jitter;
            
            const finalX = Math.max(0, Math.min(width - 1, jitterX));
            const finalY = Math.max(0, Math.min(height - 1, jitterY));
            
            const i = (finalY * width + finalX) * 4;
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            const a = pixels[i + 3];

            if (a > 50) { // Alpha threshold
                const brightness = (r + g + b) / (3 * 255);
                const posX = (finalX / width - 0.5) * scale;
                const posY = -(finalY / height - 0.5) * scale;
                const posZ = (brightness - 0.5) * depth;

                newPositions.push(new THREE.Vector3(posX, posY, posZ));
                newColors.push(new THREE.Color(r / 255, g / 255, b / 255));
            }
        }
        if (newPositions.length >= maxNodes) break;
    }
    
    return { positions: newPositions, colors: newColors };
};


const ThreeCanvas: React.FC<ThreeCanvasProps> = ({ metrics, aiThought, imageData, settings }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  
  const initialNodeCount = Math.floor(MAX_NODES * settings.particleCount);
  const initialEmbedding = createVisualEmbedding("A silent nebula awaits a spark of inquiry.", initialNodeCount, settings.colorSaturation);
  const targetPositionsRef = useRef<THREE.Vector3[]>(initialEmbedding.positions);
  const targetColorsRef = useRef<THREE.Color[]>(initialEmbedding.colors);
  const activeNodesRef = useRef<number>(initialNodeCount);

  const metricsRef = useRef(metrics);
  useEffect(() => { metricsRef.current = metrics; }, [metrics]);

  const imageDataRef = useRef(imageData);
  useEffect(() => { imageDataRef.current = imageData; }, [imageData]);
  
  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);


  useEffect(() => {
    const numNodes = Math.floor(MAX_NODES * settings.particleCount);
    if (imageData) {
        const img = new Image();
        img.onload = () => {
            const { positions, colors } = createImageEmbedding(img, numNodes);
            if (positions.length > 0) {
                targetPositionsRef.current = positions;
                targetColorsRef.current = colors;
                activeNodesRef.current = positions.length;
            }
        };
        img.onerror = () => {
            const { positions, colors } = createVisualEmbedding(aiThought, numNodes, settings.colorSaturation);
            targetPositionsRef.current = positions;
            targetColorsRef.current = colors;
            activeNodesRef.current = numNodes;
        }
        img.src = `data:image/png;base64,${imageData}`;
    } else {
        const { positions, colors } = createVisualEmbedding(aiThought, numNodes, settings.colorSaturation);
        targetPositionsRef.current = positions;
        targetColorsRef.current = colors;
        activeNodesRef.current = numNodes;
    }
  }, [imageData, aiThought, settings.particleCount, settings.colorSaturation]);

  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.FogExp2(0x000000, 0.001);
    const camera = new THREE.PerspectiveCamera(75, mount.clientWidth / mount.clientHeight, 0.1, 2000);
    camera.position.z = 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.autoClearColor = false; // Disable auto-clearing to create trails
    mount.appendChild(renderer.domElement);
    
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 100;
    controls.maxDistance = 1200;

    // Background Stars
    const starGeometry = new THREE.BufferGeometry();
    const starVertices = [];
    for (let i = 0; i < 20000; i++) {
        const x = (Math.random() - 0.5) * 2500;
        const y = (Math.random() - 0.5) * 2500;
        const z = (Math.random() - 0.5) * 2500;
        starVertices.push(x, y, z);
    }
    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
    const starMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.8, transparent: true, opacity: 0.7 });
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // Setup for the fading effect
    const fadeScene = new THREE.Scene();
    const fadeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fadeMaterial = new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
    });
    const fadePlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fadeMaterial);
    fadeScene.add(fadePlane);
    
    const geometry = new THREE.SphereGeometry(1.5, 8, 8);
    const material = new THREE.MeshBasicMaterial();
    const group = new THREE.Group();
    scene.add(group);
    
    const nodes: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>[] = [];
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
        node.scale.set(0, 0, 0); // Start hidden
        nodes.push(node);
        group.add(node);
    }
    
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(Infinity, Infinity);
    const attractorRef = React.createRef<THREE.Vector3 | null>();
    attractorRef.current = null;

    let time = 0;
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      time += 0.01;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(nodes);

      if (intersects.length > 0) {
        attractorRef.current = intersects[0].object.position;
      } else {
        attractorRef.current = null;
      }
      
      const currentSettings = settingsRef.current;
      const currentMetrics = metricsRef.current;
      group.rotation.x += 0.0001 * currentMetrics.arousal * currentSettings.movementSpeed;
      group.rotation.y += 0.0002 * currentMetrics.arousal * currentSettings.movementSpeed;

      stars.visible = currentSettings.showStars;
      if (stars.visible) {
        stars.rotation.y += 0.00005;
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
        const targetScale = isActive ? 1.0 : 0.0;
        node.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), scaleLerp);

        if (isActive) {
            const baseTargetPos = targetPositions[i];
            if (!baseTargetPos) return; // Skip if no target position

            // "4D" effect: Add a time-based oscillation to create a dynamic target
            const fourD_amplitude = 5 + currentMetrics.confusion * 30; // How far it moves
            const fourD_speed = (0.3 + currentMetrics.arousal * 1.0) * currentSettings.movementSpeed;     // How fast it moves

            // Using the node's index 'i' gives each a unique phase, preventing synchronized movement
            // Create a more complex, layered, organic motion
            const time_i = time + i * 0.31;
            const offsetVector = new THREE.Vector3(
                (Math.sin(time_i * fourD_speed) + Math.cos(time_i * fourD_speed * 0.41)) * 0.7,
                (Math.cos(time_i * fourD_speed * 0.63) + Math.sin(time_i * fourD_speed * 0.83)) * 0.7,
                (Math.sin(time_i * fourD_speed * 1.13) + Math.cos(time_i * fourD_speed * 0.57)) * 0.7
            );
            offsetVector.multiplyScalar(fourD_amplitude);
            
            // The dynamic target is the structural position plus the 4D offset
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
                node.material.color.lerp(targetColor, colorLerp);
            }
        }
      });

      controls.update();
      
      // Control trail length with arousal
      // High arousal = low opacity = longer trails
      const fadeOpacity = currentSettings.showTrails ? (0.4 - currentMetrics.arousal * 0.35) : 1.0;
      fadeMaterial.opacity = Math.max(0.05, Math.min(1.0, fadeOpacity));
      
      // Render the semi-transparent fade plane, then the main scene
      renderer.render(fadeScene, fadeCamera);
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };

    const handleMouseMove = (event: MouseEvent) => {
        const rect = mount.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      cancelAnimationFrame(animationFrameId);
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

export default ThreeCanvas;
