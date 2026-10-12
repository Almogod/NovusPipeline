/**
 * Scroll choreography state.
 *
 * GSAP timelines (useScrollChoreography) write these values; the R3F scene reads
 * them every rendered frame. The object lives outside React on purpose: scrolling
 * must never re-render a component, it only mutates numbers and requests a frame.
 */
export const choreo = {
  intro: 0, // hero load: the AST grows from its root
  scan: 0, // rule engine: a scan plane sweeps the tree and finds the smells
  morph: 0, // GNN: the tree regroups into per-function clusters
  pulse: 0.35, // message passing along edges
  propose: 0, // LLM: proposals pending on smelly nodes
  collapse: 0, // the graph is pulled into the sandbox core
  glass: 0, // glass cells assembled around the core
  part: 0, // cells parted and door open so the camera can pass
  seal: 0, // sandbox sealed
  verify: 0, // tests running inside the sandbox
  heal: 0, // tests passed: smells are fixed
  emerge: 0, // the graph leaves as the clean, modernized tree
  labels: 1, // AST label visibility

  camX: 0,
  camY: 0.6,
  camZ: 14,
  lookY: 0,
  fov: 35,
  lens: 1, // 1 = stage shifted right of the text column, 0 = centered
  orbit: 0, // rotation of the code graph around its vertical axis
}

// Handy for tuning the choreography from the console during development.
if (import.meta.env?.DEV) window.__choreo = choreo

/** Camera and stage keyframes shared by consecutive timelines, so each one starts where the last ended. */
export const KEYS = {
  hero: { camY: 0.6, camZ: 14, orbit: 0, lens: 1, fov: 35, lookY: 0 },
  rules: { camY: 0.25, camZ: 11.6, orbit: 0.35 },
  gnn: { camY: 0.95, camZ: 13.4, orbit: 0.95 },
  llm: { orbit: 1.3 },
  approach: { camY: 0.2, camZ: 9.2, lens: 0.6, orbit: 1.6 },
  inside: { camY: 0, camZ: 0.5, fov: 72, lens: 0 },
  standards: { camY: 1.5, camZ: 13.5, fov: 35, orbit: 2.1, lens: 1 },
  install: { camY: 1.2, camZ: 17, lens: 1.55, orbit: 2.7 },
}
