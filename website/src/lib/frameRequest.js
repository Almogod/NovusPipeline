/**
 * Bridge between the DOM side (GSAP) and the lazily loaded canvas.
 * Importing invalidate() from @react-three/fiber here would pull three.js into
 * the first-load bundle, so the canvas registers its invalidate on mount instead.
 */
let requester = () => {}

export function requestFrame() {
  requester()
}

export function setFrameRequester(fn) {
  requester = fn
  return () => {
    if (requester === fn) requester = () => {}
  }
}
