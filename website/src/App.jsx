import { lazy, Suspense, useRef } from 'react'
import { MotionConfig } from 'motion/react'
import { ErrorBoundary } from './components/ErrorBoundary'
import { StaticBackdrop } from './components/StaticBackdrop'
import { useScrollChoreography } from './lib/useScrollChoreography'
import { Nav } from './sections/Nav'
import { Hero } from './sections/Hero'
import { Intelligence } from './sections/Intelligence'
import { Sandbox } from './sections/Sandbox'
import { Standards } from './sections/Standards'
import { Install } from './sections/Install'

// three.js and the scene load in their own chunk; the text is readable before WebGL is ready.
const Experience = lazy(() => import('./scene/Experience'))

/**
 * Layered architecture:
 *   3D layer   fixed <Canvas> behind everything (scene/Experience)
 *   Binding    GSAP ScrollTrigger timelines -> choreography state (lib/useScrollChoreography)
 *   UI layer   HTML sections scrolling over the canvas, Motion for UI transitions
 */
export default function App() {
  const main = useRef(null)
  useScrollChoreography(main)

  return (
    <MotionConfig reducedMotion="user">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <ErrorBoundary fallback={<StaticBackdrop />}>
        <Suspense fallback={<StaticBackdrop />}>
          <Experience />
        </Suspense>
      </ErrorBoundary>
      <Nav />
      <main id="main" ref={main} className="relative z-10">
        <Hero />
        <Intelligence />
        <Sandbox />
        <Standards />
        <Install />
      </main>
    </MotionConfig>
  )
}
