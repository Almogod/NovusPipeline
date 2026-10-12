import { useMemo, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { Preload } from '@react-three/drei'
import { buildGraph } from '../lib/graph'
import { StaticBackdrop } from '../components/StaticBackdrop'
import { CameraRig } from './CameraRig'
import { CodeGraph } from './CodeGraph'
import { GlassSandbox } from './GlassSandbox'
import { RenderDriver } from './RenderDriver'
import { SceneEnvironment } from './SceneEnvironment'

function detectQuality() {
  const small = window.matchMedia('(max-width: 767px)').matches
  const coarse = window.matchMedia('(pointer: coarse)').matches
  const weakCpu = (navigator.hardwareConcurrency || 4) < 4
  return small || coarse || weakCpu ? 'low' : 'high'
}

const BACKGROUND = '#070b17'

/**
 * The fixed 3D layer behind the page. HTML scrolls over it; GSAP drives it.
 * Pointer events come from #root (eventSource), so the scene still reacts to
 * hover even though the text layer sits on top of the canvas.
 */
export default function Experience() {
  const quality = useMemo(detectQuality, [])
  const graph = useMemo(() => buildGraph({ detail: quality === 'high' ? 1 : 0.55 }), [quality])
  // Fixed layer for the AST labels. drei <Html> would otherwise portal into the event
  // source (#root), which scrolls with the page.
  const labelLayer = useRef(null)

  return (
    <div className="pointer-events-none fixed inset-0 z-0" aria-hidden="true">
      <Canvas
        frameloop="demand"
        dpr={[1, 2]}
        shadows="percentage"
        camera={{ fov: 35, near: 0.05, far: 80, position: [0, 0.6, 14] }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        eventSource={document.getElementById('root')}
        eventPrefix="client"
        fallback={<StaticBackdrop />}
        onCreated={({ gl }) => {
          // Transmission renders the scene behind the glass into its own target; halve it on small GPUs.
          gl.transmissionResolutionScale = quality === 'high' ? 1 : 0.5
        }}
      >
        <color attach="background" args={[BACKGROUND]} />
        <fog attach="fog" args={[BACKGROUND, 17, 36]} />

        <ambientLight intensity={0.3} color="#b9c9ff" />
        <directionalLight
          castShadow
          position={[4, 10, 6]}
          intensity={1.5}
          color="#ffd9b3"
          shadow-mapSize={[1024, 1024]}
          shadow-bias={-0.0004}
          shadow-normalBias={0.02}
          shadow-camera-left={-6}
          shadow-camera-right={6}
          shadow-camera-top={6}
          shadow-camera-bottom={-6}
          shadow-camera-near={1}
          shadow-camera-far={30}
        />
        <SceneEnvironment />

        <CameraRig />
        <CodeGraph graph={graph} labelLayer={labelLayer} />
        <GlassSandbox quality={quality} />

        {/* Invisible floor that only shows the shadows cast by the tree and the glass. */}
        <mesh rotation-x={-Math.PI / 2} position-y={-2.75} receiveShadow>
          <planeGeometry args={[40, 40]} />
          <shadowMaterial transparent opacity={0.32} />
        </mesh>

        <RenderDriver />
        {/* Compile every material up front, so the sandbox doesn't hitch when it first appears. */}
        <Preload all />
      </Canvas>
      <div ref={labelLayer} className="absolute inset-0 overflow-hidden" />
    </div>
  )
}
