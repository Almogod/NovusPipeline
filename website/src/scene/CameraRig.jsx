import { useFrame } from '@react-three/fiber'
import { choreo } from '../lib/choreography'

const LENS_SHIFT_MM = 7.5 // horizontal film offset that moves the stage beside the text column
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v)

/**
 * Applies the choreographed camera every frame. Responsive framing lives here,
 * so the GSAP timelines stay identical across breakpoints:
 * - wide screens shift the image right with a lens shift (no perspective change),
 * - portrait screens pull the camera back so the tree still fits.
 */
export function CameraRig() {
  useFrame(({ camera, size }) => {
    const c = choreo
    const aspect = size.width / size.height
    const wide = size.width >= 1024
    const fit = aspect < 1 ? 1 + (1 - aspect) * 1.1 : 1
    // Only pull back when far away; inside the sandbox the camera must stay put.
    const z = c.camZ * (1 + (fit - 1) * clamp01((c.camZ - 2) / 6))

    camera.position.set(c.camX, c.camY + (fit - 1) * 0.8 * clamp01((c.camZ - 2) / 6), z)
    camera.lookAt(0, c.lookY, 0)
    camera.fov = c.fov
    camera.filmOffset = wide ? -c.lens * LENS_SHIFT_MM : 0
    camera.updateProjectionMatrix()
  }, -2)
  return null
}
