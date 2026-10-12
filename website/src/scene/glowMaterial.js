import * as THREE from 'three'

/**
 * Additive, world-sized soft points: the glow around nodes and the message pulses.
 * Cheaper than a bloom pass and needs no post-processing.
 * Attributes: position, aColor (vec3), aSize (world units), aAlpha.
 */
export function createGlowMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      // Pixels per world unit at distance 1: viewportHeightPx / (2 * tan(fov / 2)).
      uProjScale: { value: 800 },
    },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aAlpha;
      uniform float uProjScale;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = min(aSize * uProjScale / max(-mv.z, 0.05), 320.0);
        vColor = aColor;
        vAlpha = aAlpha;
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float falloff = pow(max(0.0, 1.0 - d), 2.2);
        float core = smoothstep(0.32, 0.0, d) * 0.6;
        gl_FragColor = vec4(vColor * (falloff + core) * vAlpha, 1.0);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
}
