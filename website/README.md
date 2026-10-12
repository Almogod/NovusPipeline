# NovusPipeline website

Landing page for [NovusPipeline](../README.md). React + Vite + Tailwind CSS, a
React Three Fiber scene driven by GSAP ScrollTrigger, Motion for UI transitions.

```sh
npm install
npm run dev       # http://localhost:5173
npm run build     # static site in dist/, deployable to any sub-path (base: './')
npm run preview
```

## How it fits together

| Layer | Files | Role |
|---|---|---|
| 3D | `src/scene/` | Fixed `<Canvas>` behind the page (`frameloop="demand"`, `dpr={[1, 2]}`), lazy-loaded in its own chunk |
| Binding | `src/lib/useScrollChoreography.js` | One master GSAP timeline; its playhead is the summed progress of consecutive ScrollTrigger segments, so the scene is a pure function of scroll |
| State | `src/lib/choreography.js` | Plain object GSAP writes and `useFrame` reads; scrolling never re-renders React |
| UI | `src/sections/`, `src/playground/` | HTML sections over the canvas |

The scene tells the product's story in one take: a procedural Python AST
(`src/lib/graph.js`, labels use the rule engine's real rewrites) is scanned by the
rules, regrouped into per-function clusters for the GNN, pulled into the glass
sandbox where the camera enters and the tests run, and leaves as the same tree
with its smells fixed.

Rendering is on demand. Scroll updates request single frames; idle rotation and
pulses run only while someone interacts, then ease to rest and the GPU stops
(`src/scene/RenderDriver.jsx`). `prefers-reduced-motion` disables ambient motion.

`window.__choreo` exposes the choreography state in dev builds for tuning.

## Content

Copy, numbers and commands come from the repository's README, `docs/` and
`CHANGELOG.md`. The standards playground (`src/playground/simulate.js`) models the
real rule semantics of `.novus/standards/*.toml` on a sample file; it is labeled
as a simulated run. Update both when the product changes.

## Credits

`public/hdri/venice_sunset_1k.hdr`: Venice Sunset by Poly Haven (CC0), the same
file drei's `preset="sunset"` streams from a CDN, self-hosted for reliability.
Fonts: Archivo and IBM Plex Mono (SIL OFL) via Fontsource.
