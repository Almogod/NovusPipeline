import { useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { ambient, ambientWanted, markActivity, onReducedMotionChange } from '../lib/ambient'
import { setFrameRequester } from '../lib/frameRequest'

const EVENTS = ['pointermove', 'pointerdown', 'wheel', 'keydown', 'touchstart', 'scroll']

/**
 * On-demand rendering. The canvas runs with frameloop="demand", so nothing draws
 * unless something asks for a frame:
 * - scroll timelines call invalidate() on every update,
 * - this driver keeps frames coming only while ambient motion is alive, and lets
 *   it ease to rest after the user goes idle; then the GPU stops entirely.
 */
export function RenderDriver() {
  const invalidate = useThree((state) => state.invalidate)

  // Let the DOM-side scroll timelines request frames from this canvas.
  useEffect(() => setFrameRequester(() => invalidate()), [invalidate])

  useEffect(() => {
    const wake = () => {
      markActivity()
      invalidate()
    }
    EVENTS.forEach((type) => window.addEventListener(type, wake, { passive: true }))
    document.addEventListener('visibilitychange', wake)
    const stopWatching = onReducedMotionChange(wake)
    wake()
    return () => {
      EVENTS.forEach((type) => window.removeEventListener(type, wake))
      document.removeEventListener('visibilitychange', wake)
      stopWatching()
    }
  }, [invalidate])

  useFrame((_, rawDelta) => {
    const wanted = ambientWanted()
    const dt = Math.min(rawDelta, 1 / 20)
    // Ease in quickly, ease out slowly so idle rotation settles instead of stopping dead.
    const rate = wanted ? 2.5 : 0.9
    ambient.level += ((wanted ? 1 : 0) - ambient.level) * Math.min(1, dt * rate)
    if (!wanted && ambient.level < 0.002) ambient.level = 0
    if (wanted || ambient.level > 0) invalidate()
  }, -3)

  return null
}
