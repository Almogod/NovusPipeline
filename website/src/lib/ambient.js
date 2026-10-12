/**
 * Ambient motion budget.
 *
 * Idle rotation, message pulses and shimmer only run while someone is interacting
 * with the page (or did so recently). After IDLE_AFTER_MS without input the scene
 * eases to rest and the canvas stops rendering until the next scroll, pointer move
 * or key press. Users who prefer reduced motion get no ambient motion at all.
 */
const IDLE_AFTER_MS = 12000

const reducedMotionQuery =
  typeof window !== 'undefined' ? window.matchMedia('(prefers-reduced-motion: reduce)') : null

export const ambient = {
  /** 0..1, eased toward `wanted()`; every time-based animation is scaled by it. */
  level: 0,
  lastActivity: typeof performance !== 'undefined' ? performance.now() : 0,
}

export function prefersReducedMotion() {
  return Boolean(reducedMotionQuery?.matches)
}

export function ambientWanted(now = performance.now()) {
  return !prefersReducedMotion() && !document.hidden && now - ambient.lastActivity < IDLE_AFTER_MS
}

export function markActivity() {
  ambient.lastActivity = performance.now()
}

export function onReducedMotionChange(callback) {
  reducedMotionQuery?.addEventListener('change', callback)
  return () => reducedMotionQuery?.removeEventListener('change', callback)
}
