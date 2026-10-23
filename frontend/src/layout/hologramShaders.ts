import * as THREE from 'three';

/**
 * Creates a Fresnel rim-glow shader material for the holographic body envelope.
 *
 * Characteristics:
 * - Center is almost completely transparent (alpha ~0.08) so internal tumors/nerves are visible.
 * - Grazing edges glow brightly (alpha ~0.7) in cyber-cyan, establishing anatomical silhouettes.
 * - depthWrite: false prevents alpha-sorting clipping glitches in WebGL.
 */
export function createHologramFresnelMaterial(colorHex = 0x38bdf8, rimPower = 2.4): THREE.ShaderMaterial {
  const color = new THREE.Color(colorHex);

  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color },
      uRimPower: { value: rimPower },
      uBaseAlpha: { value: 0.08 },
      uRimIntensity: { value: 0.75 },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewPosition;

      void main() {
        vNormal = normalize(normalMatrix * normal);
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uRimPower;
      uniform float uBaseAlpha;
      uniform float uRimIntensity;

      varying vec3 vNormal;
      varying vec3 vViewPosition;

      void main() {
        vec3 normal = normalize(vNormal);
        vec3 viewDir = normalize(vViewPosition);

        // Fresnel term: 0 at normal, 1 at grazing silhouettes
        float fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
        fresnel = pow(fresnel, uRimPower);

        float alpha = uBaseAlpha + fresnel * uRimIntensity;
        vec3 glowColor = uColor + vec3(0.15) * fresnel;

        gl_FragColor = vec4(glowColor, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
}

/**
 * Creates a translucent golden/amber surgical safety margin expansion shell material.
 */
export function createMarginShellMaterial(colorHex = 0xfbbf24, opacity = 0.32): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: colorHex,
    emissive: 0xd97706,
    emissiveIntensity: 0.3,
    transparent: true,
    opacity: opacity,
    roughness: 0.4,
    metalness: 0.1,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

