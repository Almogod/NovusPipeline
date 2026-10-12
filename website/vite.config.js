import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // Relative asset paths, so the build works from any sub-path (e.g. GitHub Pages project sites).
  base: './',
  plugins: [react(), tailwindcss()],
  build: {
    // three.js is lazy-loaded in its own chunk; it is expected to be large.
    chunkSizeWarningLimit: 1600,
  },
})
