import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'
import { choreo } from '../lib/choreography'
import { ambient } from '../lib/ambient'

const SEALED_PITCH = 1.08 // 1-unit cells with a hairline gap
const OPEN_PITCH = 1.72 // wide enough for the camera to pass between cells
const SODIUM = new THREE.Color('#ffa53d')
const GLACIER = new THREE.Color('#8bd3ff')

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v)
const segment = (p, delay, span) => clamp01((p - delay) / span)
const easeOut = (t) => 1 - Math.pow(1 - t, 3)

// The 12 edges of a unit cube as 24 segment endpoints.
const CUBE_EDGES = (() => {
  const corners = []
  for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) corners.push([x, y, z])
  const out = []
  corners.forEach((a, i) =>
    corners.forEach((b, j) => {
      const differing = a.filter((v, k) => v !== b[k]).length
      if (j > i && differing === 1) out.push(new THREE.Vector3(...a), new THREE.Vector3(...b))
    }),
  )
  return out
})()

/**
 * The test sandbox: a 3x3x3 matrix of glass cells with the code at its center.
 * The front-center cell is the door: it shrinks away so the camera can enter,
 * then closes behind it when the sandbox seals.
 */
export function GlassSandbox({ quality }) {
  const root = useRef(null)
  const cellMesh = useRef(null)
  const cage = useRef(null)
  const verifyRing = useRef(null)
  const light = useRef(null)

  const cells = useMemo(() => {
    const list = []
    let seed = 3
    const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646
    for (let x = -1; x <= 1; x++)
      for (let y = -1; y <= 1; y++)
        for (let z = -1; z <= 1; z++) {
          if (!x && !y && !z) continue // the core lives here
          const ring = Math.abs(x) + Math.abs(y) + Math.abs(z)
          list.push({
            dir: new THREE.Vector3(x, y, z),
            door: x === 0 && y === 0 && z === 1,
            delay: (ring - 1) * 0.15 + rand() * 0.12,
            spin: new THREE.Euler((rand() - 0.5) * 3, (rand() - 0.5) * 3, (rand() - 0.5) * 3),
          })
        }
    return list
  }, [])

  const geometry = useMemo(() => new RoundedBoxGeometry(1, 1, 1, 5, 0.085), [])
  const edgeGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(cells.length * CUBE_EDGES.length * 3), 3).setUsage(
        THREE.DynamicDrawUsage,
      ),
    )
    return geo
  }, [cells])
  const edgeMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: '#8bd3ff',
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )

  useEffect(
    () => () => {
      geometry.dispose()
      edgeGeometry.dispose()
      edgeMaterial.dispose()
    },
    [geometry, edgeGeometry, edgeMaterial],
  )

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const vertex = useMemo(() => new THREE.Vector3(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const time = useRef(0)

  useFrame((_, rawDelta) => {
    const c = choreo
    color.copy(SODIUM).lerp(GLACIER, c.heal)
    light.current.color.copy(color)
    light.current.intensity = c.glass * c.collapse * (1.6 + 2.4 * c.verify)
    const visible = c.glass > 0.001
    root.current.visible = visible
    if (!visible) return
    time.current += Math.min(rawDelta, 1 / 20) * ambient.level

    const pitch = SEALED_PITCH + (OPEN_PITCH - SEALED_PITCH) * c.part
    const edgePositions = edgeGeometry.attributes.position.array
    cells.forEach((cell, index) => {
      const appear = easeOut(segment(c.glass, cell.delay, 0.55))
      const scale = appear * (cell.door ? 1 - easeOut(c.part) : 1)
      const distance = pitch + (1 - appear) * 2.6 // cells fly in from outside
      dummy.position.copy(cell.dir).multiplyScalar(distance)
      dummy.rotation.set(cell.spin.x * (1 - appear), cell.spin.y * (1 - appear), cell.spin.z * (1 - appear))
      dummy.scale.setScalar(Math.max(scale, 1e-4))
      dummy.updateMatrix()
      cellMesh.current.setMatrixAt(index, dummy.matrix)

      const offset = index * CUBE_EDGES.length * 3
      CUBE_EDGES.forEach((corner, v) => {
        vertex.copy(corner).applyMatrix4(dummy.matrix)
        edgePositions[offset + v * 3] = vertex.x
        edgePositions[offset + v * 3 + 1] = vertex.y
        edgePositions[offset + v * 3 + 2] = vertex.z
      })
    })
    cellMesh.current.instanceMatrix.needsUpdate = true
    edgeGeometry.attributes.position.needsUpdate = true
    edgeMaterial.opacity = c.glass * (0.06 + 0.2 * c.seal)

    // The test cage around the code: sodium while tests run, glacier once they pass.
    const cageScale = 0.3 * easeOut(c.seal)
    cage.current.visible = cageScale > 0.001
    cage.current.scale.setScalar(Math.max(cageScale, 1e-4))
    cage.current.rotation.set(time.current * 0.4, time.current * 0.55, 0)
    cage.current.material.color.copy(color)
    cage.current.material.opacity = 0.45 * c.seal

    // A ring sweeps over the core while the tests run.
    const sweep = Math.sin(Math.PI * clamp01(c.verify))
    verifyRing.current.visible = sweep > 0.002
    verifyRing.current.position.y = -0.28 + 0.56 * c.verify
    verifyRing.current.material.opacity = sweep * 0.9

  })

  return (
    <>
      {/* Outside the toggled group: a changing light count would force shader recompiles mid-scroll. */}
      <pointLight ref={light} distance={6} decay={1.6} intensity={0} />
      <group ref={root} visible={false}>
      <instancedMesh ref={cellMesh} args={[geometry, undefined, cells.length]} castShadow frustumCulled={false}>
        <meshPhysicalMaterial
          color="#e8f3ff"
          transmission={1}
          // Thin, slightly frosted walls: a thick slab magnifies the core behind it into a confusing second copy.
          thickness={0.3}
          roughness={quality === 'high' ? 0.16 : 0.22}
          ior={1.3}
          attenuationColor="#a3dcff"
          attenuationDistance={2.6}
          specularIntensity={1}
          clearcoat={0.5}
          clearcoatRoughness={0.08}
          envMapIntensity={1.4}
          // Glows behind the glass stay visible through it.
          depthWrite={false}
        />
      </instancedMesh>
      <lineSegments geometry={edgeGeometry} material={edgeMaterial} frustumCulled={false} />

      <mesh ref={cage}>
        <icosahedronGeometry args={[1, 1]} />
        <meshBasicMaterial wireframe transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      <mesh ref={verifyRing} rotation-x={-Math.PI / 2}>
        <ringGeometry args={[0.24, 0.255, 96]} />
        <meshBasicMaterial
          color="#8bd3ff"
          transparent
          depthWrite={false}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
    </>
  )
}
