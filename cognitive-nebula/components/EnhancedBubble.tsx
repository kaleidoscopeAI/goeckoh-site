import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

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
  zcr?: number;
}

interface EnhancedBubbleProps {
  bubbleState: BubbleState;
  voiceData?: {
    waveform?: Float32Array;
    spectrum?: Float32Array;
    f0?: number;
  };
  settings?: {
    enableBloom?: boolean;
    enableParticles?: boolean;
    enableVolumetric?: boolean;
    particleCount?: number;
  };
}

// Custom shader for bubble surface with real-time deformation
const bubbleVertexShader = `
  uniform float uTime;
  uniform float uSpike;
  uniform float uEnergy;
  uniform float uRadius;
  uniform float uF0;
  
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  
  // Noise function for organic deformation
  float noise(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
  }
  
  // Smooth noise
  float smoothNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = noise(i);
    float b = noise(i + vec3(1.0, 0.0, 0.0));
    float c = noise(i + vec3(0.0, 1.0, 0.0));
    float d = noise(i + vec3(1.0, 1.0, 0.0));
    
    float e = noise(i + vec3(0.0, 0.0, 1.0));
    float f_n = noise(i + vec3(1.0, 0.0, 1.0));
    float g = noise(i + vec3(0.0, 1.0, 1.0));
    float h = noise(i + vec3(1.0, 1.0, 1.0));
    
    float ab = mix(a, b, f.x);
    float cd = mix(c, d, f.x);
    float ef = mix(e, f_n, f.x);
    float gh = mix(g, h, f.x);
    
    float abcd = mix(ab, cd, f.y);
    float efgh = mix(ef, gh, f.y);
    
    return mix(abcd, efgh, f.z);
  }
  
  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    
    vec3 pos = position;
    
    // Base radius scaling
    float baseScale = uRadius;
    
    // Energy-based expansion
    float energyWave = sin(uTime * 2.0 + length(pos) * 5.0) * uEnergy * 0.1;
    
    // Spike deformation (Bouba/Kiki effect)
    float spikeAmount = uSpike;
    vec3 spikeDir = normalize(pos);
    float spikePattern = smoothNoise(pos * 3.0 + uTime * 0.5) * 2.0 - 1.0;
    pos += spikeDir * spikePattern * spikeAmount * 0.15;
    
    // Frequency-based ripple
    float f0Norm = (uF0 - 80.0) / 320.0;
    float ripple = sin(length(pos) * 10.0 - uTime * f0Norm * 5.0) * 0.05;
    pos += normalize(pos) * ripple * uEnergy;
    
    // Final position
    pos *= baseScale + energyWave;
    vPosition = pos;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const bubbleFragmentShader = `
  uniform vec3 uColor;
  uniform float uRoughness;
  uniform float uMetalness;
  uniform float uTime;
  uniform float uEnergy;
  uniform float uSpike;
  
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  
  // Fresnel effect for edge glow
  float fresnel(vec3 viewDir, vec3 normal) {
    return pow(1.0 - dot(viewDir, normal), 2.0);
  }
  
  void main() {
    vec3 viewDir = normalize(cameraPosition - vPosition);
    vec3 normal = normalize(vNormal);
    
    // Base color with energy modulation
    vec3 baseColor = uColor;
    baseColor += vec3(uEnergy * 0.3) * (1.0 - uSpike);
    
    // Fresnel edge glow
    float fresnelFactor = fresnel(viewDir, normal);
    vec3 edgeGlow = baseColor * fresnelFactor * (1.0 + uEnergy * 2.0);
    
    // Metallic reflection
    vec3 metallicColor = mix(baseColor, vec3(1.0), uMetalness * 0.5);
    
    // Roughness affects specular
    float specular = pow(max(dot(viewDir, reflect(-viewDir, normal)), 0.0), 
                         mix(32.0, 4.0, uRoughness));
    
    vec3 finalColor = mix(metallicColor, baseColor, uRoughness);
    finalColor += edgeGlow * 0.5;
    finalColor += specular * uMetalness * 0.3;
    
    // Energy-based intensity
    finalColor *= 1.0 + uEnergy * 0.5;
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

const EnhancedBubble: React.FC<EnhancedBubbleProps> = ({
  bubbleState,
  voiceData,
  settings = {}
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const {
    enableBloom = true,
    enableParticles = true,
    enableVolumetric = true,
    particleCount = 5000
  } = settings;

  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.FogExp2(0x000000, 0.0008);

    const camera = new THREE.PerspectiveCamera(
      50,
      mount.clientWidth / mount.clientHeight,
      0.1,
      2000
    );
    camera.position.set(0, 0, 8);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
      alpha: true
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    mount.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 3;
    controls.maxDistance = 20;

    // Bubble geometry - high resolution for smooth deformation
    const bubbleGeometry = new THREE.IcosahedronGeometry(1, 4);
    const bubbleMaterial = new THREE.ShaderMaterial({
      vertexShader: bubbleVertexShader,
      fragmentShader: bubbleFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uSpike: { value: bubbleState.spike },
        uEnergy: { value: bubbleState.energy || 0 },
        uRadius: { value: bubbleState.radius },
        uF0: { value: bubbleState.f0 || 220 },
        uColor: { value: new THREE.Color(bubbleState.color_r, bubbleState.color_g, bubbleState.color_b) },
        uRoughness: { value: bubbleState.rough },
        uMetalness: { value: bubbleState.metal },
        cameraPosition: { value: camera.position }
      },
      side: THREE.DoubleSide
    });

    const bubble = new THREE.Mesh(bubbleGeometry, bubbleMaterial);
    scene.add(bubble);

    // Volumetric light rays
    let volumetricGroup: THREE.Group | null = null;
    if (enableVolumetric) {
      volumetricGroup = new THREE.Group();
      const rayCount = 32;
      for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const rayGeometry = new THREE.ConeGeometry(0.02, 2, 8, 1, true);
        const rayMaterial = new THREE.MeshBasicMaterial({
          color: new THREE.Color(bubbleState.color_r, bubbleState.color_g, bubbleState.color_b),
          transparent: true,
          opacity: 0.1,
          side: THREE.DoubleSide
        });
        const ray = new THREE.Mesh(rayGeometry, rayMaterial);
        ray.position.set(
          Math.cos(angle) * 1.5,
          Math.sin(angle) * 1.5,
          0
        );
        ray.lookAt(0, 0, 0);
        volumetricGroup.add(ray);
      }
      scene.add(volumetricGroup);
    }

    // Particle system for voice energy
    let particleSystem: THREE.Points | null = null;
    if (enableParticles) {
      const particlesGeometry = new THREE.BufferGeometry();
      const positions = new Float32Array(particleCount * 3);
      const colors = new Float32Array(particleCount * 3);
      const sizes = new Float32Array(particleCount);

      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        // Spherical distribution
        const radius = 2 + Math.random() * 3;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        
        positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
        positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        positions[i3 + 2] = radius * Math.cos(phi);
        
        colors[i3] = bubbleState.color_r;
        colors[i3 + 1] = bubbleState.color_g;
        colors[i3 + 2] = bubbleState.color_b;
        
        sizes[i] = Math.random() * 0.05 + 0.02;
      }

      particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      particlesGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

      const particlesMaterial = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true
      });

      particleSystem = new THREE.Points(particlesGeometry, particlesMaterial);
      scene.add(particleSystem);
    }

    // Advanced lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(
      new THREE.Color(bubbleState.color_r, bubbleState.color_g, bubbleState.color_b),
      1.5
    );
    keyLight.position.set(5, 5, 5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x00ffff, 0.8);
    rimLight.position.set(-5, -3, -5);
    scene.add(rimLight);

    // Point lights for dynamic illumination
    const pointLights: THREE.PointLight[] = [];
    for (let i = 0; i < 4; i++) {
      const light = new THREE.PointLight(
        new THREE.Color(bubbleState.color_r, bubbleState.color_g, bubbleState.color_b),
        1,
        10
      );
      const angle = (i / 4) * Math.PI * 2;
      light.position.set(
        Math.cos(angle) * 3,
        Math.sin(angle) * 3,
        Math.sin(angle * 2) * 2
      );
      scene.add(light);
      pointLights.push(light);
    }

    // Post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    if (enableBloom) {
      const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(mount.clientWidth, mount.clientHeight),
        1.5, // strength
        0.4, // radius
        0.85 // threshold
      );
      composer.addPass(bloomPass);
    }

    // Film grain for texture
    const filmPass = new FilmPass({
      noiseIntensity: 0.1,
      scanlinesIntensity: 0.05,
      scanlinesCount: 512,
      grayscale: false
    });
    composer.addPass(filmPass);

    // Animation
    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.016;

      // Update bubble shader uniforms
      bubbleMaterial.uniforms.uTime.value = time;
      bubbleMaterial.uniforms.uSpike.value = bubbleState.spike;
      bubbleMaterial.uniforms.uEnergy.value = bubbleState.energy || 0;
      bubbleMaterial.uniforms.uRadius.value = bubbleState.radius;
      bubbleMaterial.uniforms.uF0.value = bubbleState.f0 || 220;
      bubbleMaterial.uniforms.uColor.value.setRGB(
        bubbleState.color_r,
        bubbleState.color_g,
        bubbleState.color_b
      );
      bubbleMaterial.uniforms.uRoughness.value = bubbleState.rough;
      bubbleMaterial.uniforms.uMetalness.value = bubbleState.metal;
      bubbleMaterial.uniforms.cameraPosition.value.copy(camera.position);

      // Animate particles
      if (particleSystem) {
        const positions = particleSystem.geometry.attributes.position.array as Float32Array;
        const energy = bubbleState.energy || 0;
        
        for (let i = 0; i < particleCount; i++) {
          const i3 = i * 3;
          const radius = Math.sqrt(
            positions[i3] ** 2 + positions[i3 + 1] ** 2 + positions[i3 + 2] ** 2
          );
          const baseRadius = 2 + (i % 3) * 1;
          
          // Energy-based expansion
          const targetRadius = baseRadius + energy * 2;
          const currentRadius = radius;
          const newRadius = currentRadius + (targetRadius - currentRadius) * 0.1;
          
          const scale = newRadius / (currentRadius || 1);
          positions[i3] *= scale;
          positions[i3 + 1] *= scale;
          positions[i3 + 2] *= scale;
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
      }

      // Animate volumetric rays
      if (volumetricGroup) {
        volumetricGroup.children.forEach((ray, i) => {
          const energy = bubbleState.energy || 0;
          (ray as THREE.Mesh).material.opacity = 0.1 + energy * 0.3;
          ray.rotation.z += 0.01;
        });
      }

      // Animate point lights
      pointLights.forEach((light, i) => {
        const angle = time * 0.5 + (i / pointLights.length) * Math.PI * 2;
        light.position.x = Math.cos(angle) * 3;
        light.position.y = Math.sin(angle) * 3;
        light.intensity = 0.5 + (bubbleState.energy || 0) * 1.5;
      });

      controls.update();
      composer.render();
    };
    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      composer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mount && renderer.domElement) {
        mount.removeChild(renderer.domElement);
      }
      renderer.dispose();
      bubbleGeometry.dispose();
      bubbleMaterial.dispose();
    };
  }, [bubbleState, voiceData, settings]);

  return <div ref={mountRef} className="absolute inset-0 w-full h-full" />;
};

export default EnhancedBubble;

