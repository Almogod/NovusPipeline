import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { useSpring } from '@react-spring/three'
import * as THREE from 'three'
import { choreo } from '../lib/choreography'
import { ambient, prefersReducedMotion } from '../lib/ambient'
import { createGlowMaterial } from './glowMaterial'

const SODIUM = new THREE.Color('#ffa53d')
const GLACIER = new THREE.Color('#8bd3ff')
const PAPER = new THREE.Color('#dfe8f7')
const WHITE = new THREE.Color('#ffffff')

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v)
/** Local progress of a staggered segment that starts at `delay` and lasts `span` of the parent progress. */
const segment = (p, delay, span) => clamp01((p - delay) / span)
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
const smooth = (t) => t * t * (3 - 2 * t)

// Emissive follows each instance's color, so smells glow sodium and healthy nodes glacier.
function tintEmissiveByInstance(shader) {
  shader.fragmentShader = shader.fragmentShader.replace(
    'vec3 totalEmissiveRadiance = emissive;',
    `vec3 totalEmissiveRadiance = emissive;
    #ifdef USE_COLOR
      totalEmissiveRadiance *= vColor.rgb;
    #endif`,
  )
}

const IDLE_SPEED = 0.15 // rad/s
const HOVER_SPEED = 0.035

/**
 * The AST as an instanced low-poly node network: faceted nodes, additive halos,
 * edges and message pulses. Every frame it blends the layouts according to the
 * scroll choreography (legacy tree -> GNN clusters -> sandbox core -> modern tree).
 */
export function CodeGraph({ graph, labelLayer }) {
  const { n, layouts, scatter, depth, size, tone, smell, morphDelay, collapseDelay, phase, edges, edgeType, pulses } =
    graph

  const group = useRef(null)
  const mesh = useRef(null)
  const scanRing = useRef(null)
  const labelGroups = useRef([])
  const labelEls = useRef([])
  const spin = useRef(0)
  const time = useRef(0)

  // Physics-based idle rotation and pointer tilt (react-spring drives the values, useFrame reads them).
  const [spring, api] = useSpring(
    () => ({ tiltX: 0, tiltY: 0, speed: IDLE_SPEED, hover: 0, config: { mass: 1.4, tension: 120, friction: 32 } }),
    [],
  )

  useEffect(() => {
    const onMove = (event) => {
      if (prefersReducedMotion()) return
      const x = (event.clientX / window.innerWidth) * 2 - 1
      const y = (event.clientY / window.innerHeight) * 2 - 1
      api.start({ tiltY: x * 0.3, tiltX: y * 0.14 })
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => window.removeEventListener('pointermove', onMove)
  }, [api])

  const buffers = useMemo(() => {
    const edgeCount = edges.length / 2
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    const grow = new Float32Array(n)
    const collapsed = new Float32Array(n)

    const halo = new THREE.BufferGeometry()
    halo.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage))
    halo.setAttribute('aColor', new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage))
    halo.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage))
    halo.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage))

    const lines = new THREE.BufferGeometry()
    lines.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(edgeCount * 6), 3).setUsage(THREE.DynamicDrawUsage),
    )
    // RGBA vertex colors: three.js enables per-vertex alpha when the attribute has 4 components.
    lines.setAttribute('color', new THREE.BufferAttribute(new Float32Array(edgeCount * 8), 4).setUsage(THREE.DynamicDrawUsage))

    const pulse = new THREE.BufferGeometry()
    const p = pulses.length
    pulse.setAttribute('position', new THREE.BufferAttribute(new Float32Array(p * 3), 3).setUsage(THREE.DynamicDrawUsage))
    pulse.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(p * 3), 3).setUsage(THREE.DynamicDrawUsage))
    pulse.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(p).fill(0.1), 1))
    pulse.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array(p), 1).setUsage(THREE.DynamicDrawUsage))

    return { positions, colors, grow, collapsed, halo, lines, pulse }
  }, [n, edges, pulses])

  const glow = useMemo(() => createGlowMaterial(), [])

  useEffect(
    () => () => {
      buffers.halo.dispose()
      buffers.lines.dispose()
      buffers.pulse.dispose()
      glow.dispose()
    },
    [buffers, glow],
  )

  // Allocate instance colors before the first compile so the shader is built with them.
  useLayoutEffect(() => {
    const color = new THREE.Color()
    for (let i = 0; i < n; i++) mesh.current.setColorAt(i, color.copy(PAPER))
  }, [n])

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const screen = useMemo(() => ({ node: new THREE.Vector3(), center: new THREE.Vector3() }), [])
  const color = useMemo(() => new THREE.Color(), [])

  // Priority -1: positions are final before drei <Html> labels project themselves.
  useFrame((state, rawDelta) => {
    const dt = Math.min(rawDelta, 1 / 20)
    const amb = ambient.level
    time.current += dt * amb
    const t = time.current
    const c = choreo
    const { legacy, modern, cluster, core } = layouts
    const { positions, colors, grow, collapsed } = buffers

    spin.current += spring.speed.get() * dt * amb
    const settle = 1 - c.collapse
    group.current.rotation.set(spring.tiltX.get() * settle, spin.current + spring.tiltY.get() * settle + c.orbit, 0)
    const hover = spring.hover.get()

    // Rule-engine audit: a plane sweeps from the root down through the tree.
    const { top, bottom } = graph.bounds
    const scanY = top + 0.5 - (top - bottom + 1) * c.scan
    const sweep = Math.sin(Math.PI * clamp01(c.scan)) * (1 - c.morph)
    scanRing.current.position.y = scanY
    scanRing.current.visible = sweep > 0.002
    scanRing.current.children.forEach((child, k) => (child.material.opacity = sweep * (k === 0 ? 0.75 : 0.06)))

    const haloSize = buffers.halo.attributes.aSize.array
    const haloAlpha = buffers.halo.attributes.aAlpha.array

    for (let i = 0; i < n; i++) {
      const i3 = i * 3
      const m = easeInOut(segment(c.morph, morphDelay[i], 0.58))
      const arc = Math.sin(Math.PI * m) * 0.9 // fragments outward mid-morph
      let x = legacy[i3] + (cluster[i3] - legacy[i3]) * m + scatter[i3] * arc
      let y = legacy[i3 + 1] + (cluster[i3 + 1] - legacy[i3 + 1]) * m + scatter[i3 + 1] * arc
      let z = legacy[i3 + 2] + (cluster[i3 + 2] - legacy[i3 + 2]) * m + scatter[i3 + 2] * arc
      if (c.emerge > 0) {
        x += (modern[i3] - x) * c.emerge
        y += (modern[i3 + 1] - y) * c.emerge
        z += (modern[i3 + 2] - z) * c.emerge
      }
      const k = easeInOut(segment(c.collapse, collapseDelay[i], 0.55))
      x += (core[i3] - x) * k
      y += (core[i3 + 1] - y) * k
      z += (core[i3 + 2] - z) * k

      const wobble = 0.03 * amb * (1 - k)
      x += Math.sin(t * 0.8 + phase[i]) * wobble
      y += Math.cos(t * 0.7 + phase[i] * 1.3) * wobble
      z += Math.sin(t * 0.6 + phase[i] * 0.7) * wobble
      positions[i3] = x
      positions[i3 + 1] = y
      positions[i3 + 2] = z

      const g = smooth(clamp01(c.intro * (graph.maxDepth + 2) - depth[i]))
      grow[i] = g
      collapsed[i] = k
      const s = size[i] * g * (1 - 0.87 * k) * (1 + 0.1 * hover)
      dummy.position.set(x, y, z)
      dummy.rotation.set(phase[i] + t * 0.35, phase[i] * 1.7 + t * 0.25, 0)
      dummy.scale.setScalar(Math.max(s, 1e-4))
      dummy.updateMatrix()
      mesh.current.setMatrixAt(i, dummy.matrix)

      color.copy(PAPER).lerp(GLACIER, tone[i])
      let hot = 0
      if (smell[i]) {
        const found = c.scan >= 1 || legacy[i3 + 1] > scanY ? 1 : 0
        hot = (0.42 + 0.58 * found) * (1 - c.heal)
        color.lerp(SODIUM, hot)
        // LLM proposals pending: the node flickers between legacy and modern until the tests decide.
        const pending = c.propose * (1 - c.heal) * (0.5 + 0.5 * Math.sin(t * 4 + phase[i] * 3))
        color.lerp(GLACIER, pending * 0.75)
      }
      const flash = sweep * Math.max(0, 1 - Math.abs(legacy[i3 + 1] - scanY) / 0.35)
      color.lerp(WHITE, flash * 0.6)
      mesh.current.setColorAt(i, color)

      colors[i3] = color.r
      colors[i3 + 1] = color.g
      colors[i3 + 2] = color.b
      haloSize[i] = s * (smell[i] ? 6 : 4.4) * (1 + flash)
      haloAlpha[i] = (smell[i] ? 0.5 + 0.5 * hot : 0.3) * g * (1 - 0.55 * k)
    }
    mesh.current.instanceMatrix.needsUpdate = true
    mesh.current.instanceColor.needsUpdate = true
    for (const name of ['position', 'aColor', 'aSize', 'aAlpha']) buffers.halo.attributes[name].needsUpdate = true

    // Edges inherit their endpoints' colors; sibling edges appear with the GNN view.
    const linePos = buffers.lines.attributes.position.array
    const lineCol = buffers.lines.attributes.color.array
    for (let e = 0; e < edgeType.length; e++) {
      const a = edges[e * 2]
      const b = edges[e * 2 + 1]
      const visible = Math.min(grow[a], grow[b]) * (1 - Math.max(collapsed[a], collapsed[b]))
      const alpha = (edgeType[e] === 0 ? 0.3 : 0.24 * c.morph) * visible
      for (let v = 0; v < 2; v++) {
        const node = v === 0 ? a : b
        const p = e * 6 + v * 3
        linePos[p] = positions[node * 3]
        linePos[p + 1] = positions[node * 3 + 1]
        linePos[p + 2] = positions[node * 3 + 2]
        const q = e * 8 + v * 4
        lineCol[q] = colors[node * 3]
        lineCol[q + 1] = colors[node * 3 + 1]
        lineCol[q + 2] = colors[node * 3 + 2]
        lineCol[q + 3] = alpha
      }
    }
    buffers.lines.attributes.position.needsUpdate = true
    buffers.lines.attributes.color.needsUpdate = true

    // Pulses: data flowing parent -> child (message passing in the GNN view).
    const pulsePos = buffers.pulse.attributes.position.array
    const pulseCol = buffers.pulse.attributes.aColor.array
    const pulseAlpha = buffers.pulse.attributes.aAlpha.array
    pulses.forEach((pulse, j) => {
      const a = edges[pulse.edge * 2]
      const b = edges[pulse.edge * 2 + 1]
      const f = (t * pulse.speed + pulse.phase) % 1
      for (let d = 0; d < 3; d++) {
        pulsePos[j * 3 + d] = positions[a * 3 + d] + (positions[b * 3 + d] - positions[a * 3 + d]) * f
        pulseCol[j * 3 + d] = colors[b * 3 + d]
      }
      const visible = Math.min(grow[a], grow[b]) * (1 - Math.max(collapsed[a], collapsed[b]))
      pulseAlpha[j] = c.pulse * Math.sin(Math.PI * f) * visible * amb
    })
    for (const name of ['position', 'aColor', 'aAlpha']) buffers.pulse.attributes[name].needsUpdate = true

    glow.uniforms.uProjScale.value =
      (state.size.height * state.gl.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(state.camera.fov) / 2))

    // Labels follow their nodes and sit on the outer side of the tree; CSS variables crossfade legacy -> modern text.
    group.current.updateMatrixWorld()
    screen.center.setFromMatrixPosition(group.current.matrixWorld).project(state.camera)
    graph.labels.forEach((label, j) => {
      const anchor = labelGroups.current[j]
      const el = labelEls.current[j]
      if (!anchor) return
      const i3 = label.index * 3
      anchor.position.set(positions[i3], positions[i3 + 1], positions[i3 + 2])
      if (!el) return
      screen.node.set(positions[i3], positions[i3 + 1], positions[i3 + 2]).applyMatrix4(group.current.matrixWorld).project(state.camera)
      // Outer side of the tree, unless that would run off the right edge of the screen.
      const side = screen.node.x < screen.center.x - 0.01 || screen.node.x > 0.55 ? 'left' : 'right'
      if (el.dataset.side !== side) el.dataset.side = side
      const found = c.scan >= 1 || layouts.legacy[i3 + 1] > scanY ? 1 : 0
      el.style.opacity = (c.labels * grow[label.index] * (1 - collapsed[label.index])).toFixed(3)
      el.style.setProperty('--found', found)
      el.style.setProperty('--heal', c.heal.toFixed(3))
    })
  }, -1)

  return (
    <group ref={group}>
      <instancedMesh ref={mesh} args={[undefined, undefined, n]} castShadow frustumCulled={false}>
        <icosahedronGeometry args={[1, 0]} />
        <meshStandardMaterial
          flatShading
          metalness={0.5}
          roughness={0.28}
          emissive="#ffffff"
          emissiveIntensity={0.55}
          envMapIntensity={1.2}
          onBeforeCompile={tintEmissiveByInstance}
        />
      </instancedMesh>

      <lineSegments geometry={buffers.lines} frustumCulled={false}>
        <lineBasicMaterial vertexColors transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </lineSegments>
      <points geometry={buffers.halo} material={glow} frustumCulled={false} />
      <points geometry={buffers.pulse} material={glow} frustumCulled={false} />

      <group ref={scanRing} visible={false}>
        <mesh rotation-x={-Math.PI / 2}>
          <ringGeometry args={[scanRadius(graph) - 0.02, scanRadius(graph) + 0.02, 160]} />
          <meshBasicMaterial color="#8bd3ff" transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
        </mesh>
        <mesh rotation-x={-Math.PI / 2}>
          <circleGeometry args={[scanRadius(graph), 96]} />
          <meshBasicMaterial color="#8bd3ff" transparent depthWrite={false} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} />
        </mesh>
      </group>

      {/* Hover target for the spring: slows the idle rotation so the tree can be inspected. */}
      <mesh
        onPointerEnter={() => api.start({ speed: HOVER_SPEED, hover: 1 })}
        onPointerLeave={() => api.start({ speed: IDLE_SPEED, hover: 0 })}
      >
        <sphereGeometry args={[2.7, 16, 12]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} />
      </mesh>

      {graph.labels.map((label, j) => (
        <group key={label.key} ref={(el) => (labelGroups.current[j] = el)}>
          <Html portal={labelLayer} zIndexRange={[4, 0]} style={{ pointerEvents: 'none' }}>
            <div
              ref={(el) => (labelEls.current[j] = el)}
              className="np-label"
              data-root={label.key === 'root' ? '' : undefined}
            >
              <span className="np-code">
                <span className="legacy">{label.legacy}</span>
                {label.modern && <span className="modern">{label.modern}</span>}
              </span>
              {label.tag && (
                <span className="np-tag">
                  <span className="found">{label.tag}</span>
                  <span className="fixed">fixed</span>
                </span>
              )}
            </div>
          </Html>
        </group>
      ))}
    </group>
  )
}

const scanRadius = (graph) => graph.bounds.radius + 0.25
