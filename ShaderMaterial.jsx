import * as THREE from 'three';

const ShaderMaterial = () => {
  return new THREE.ShaderMaterial({
    uniforms: {},
    vertexShader: `
      attribute vec3 instanceColor;
      attribute float instanceIntensity;
      varying vec3 vInstanceColor;
      varying float vInstanceIntensity;
      void main() {
        vInstanceColor = instanceColor;
        vInstanceIntensity = instanceIntensity;
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vInstanceColor;
      varying float vInstanceIntensity;
      void main() {
        gl_FragColor = vec4(vInstanceColor, 1.0);
        gl_FragColor.rgb *= vInstanceIntensity; // Simple glow emulation via brightness
      }
    `,
    toneMapped: false,
    transparent: true,
    blending: THREE.AdditiveBlending,
  });
};

export default ShaderMaterial;
