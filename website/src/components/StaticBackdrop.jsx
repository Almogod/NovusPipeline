/** Shown while the 3D scene loads, and instead of it when WebGL is unavailable. */
export function StaticBackdrop() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0"
      style={{
        background:
          'radial-gradient(60rem 40rem at 72% 38%, rgb(139 211 255 / 0.09), transparent 70%), radial-gradient(30rem 22rem at 64% 62%, rgb(255 165 61 / 0.06), transparent 70%), #070b17',
      }}
    />
  )
}
