import { Component, Suspense, useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import { Environment, Lightformer } from '@react-three/drei'

// Relative to the page, so it also works when the site is served from a sub-path.
const SUNSET_HDR = './hdri/venice_sunset_1k.hdr'

/** Renders `fallback` if the HDR fails to load (offline, blocked CDN, strict CSP). */
class EnvironmentBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function RequestFrame() {
  const invalidate = useThree((state) => state.invalidate)
  useEffect(() => invalidate(), [invalidate])
  return null
}

/** Procedural studio lighting with sunset tones; needs no network. */
function StudioLights() {
  return (
    <Environment resolution={256} frames={1}>
      <Lightformer form="rect" intensity={3} color="#ffb070" position={[0, 2.5, -7]} scale={[14, 2.5, 1]} />
      <Lightformer form="rect" intensity={1.4} color="#ff7f6b" position={[-7, 0.5, -1]} rotation-y={Math.PI / 2} scale={[9, 1.5, 1]} />
      <Lightformer form="rect" intensity={1.2} color="#8bd3ff" position={[7, 1, 1]} rotation-y={-Math.PI / 2} scale={[9, 2, 1]} />
      <Lightformer form="ring" intensity={2} color="#ffffff" position={[3, 6, 4]} scale={2.5} />
      <RequestFrame />
    </Environment>
  )
}

/**
 * Image-based lighting for metal and glass reflections: the HDR behind drei's
 * "sunset" preset (Venice Sunset by Poly Haven, CC0), self-hosted in public/hdri.
 * `preset="sunset"` streams the same file from a public CDN at runtime; that CDN
 * timed out during testing, so the page doesn't depend on it. If the file can't
 * load, studio lightformers take over.
 */
export function SceneEnvironment() {
  return (
    <EnvironmentBoundary fallback={<StudioLights />}>
      <Suspense fallback={null}>
        <Environment files={SUNSET_HDR} environmentIntensity={0.85} />
        <RequestFrame />
      </Suspense>
    </EnvironmentBoundary>
  )
}
