import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useGSAP } from '@gsap/react'
import { choreo, KEYS } from './choreography'
import { prefersReducedMotion } from './ambient'
import { requestFrame } from './frameRequest'

gsap.registerPlugin(ScrollTrigger, useGSAP)

const pick = (source, keys) => Object.fromEntries(keys.map((key) => [key, source[key]]))

/**
 * Binds page scroll to the 3D scene.
 *
 * Everything lives on ONE paused master timeline, so the scene is a pure function
 * of scroll position: jumps (nav links, reloads mid-page, fast flicks) render the
 * tweens in order and never leave a stale value behind, which separate
 * per-section timelines would.
 *
 * The page is cut into consecutive, non-overlapping scroll segments. Each is a
 * ScrollTrigger, and the master playhead is the sum of their progress values, so
 * segment i occupies [i, i + 1] on the timeline. Positions below are authored in
 * tenths of a segment.
 *
 * The canvas renders on demand: every timeline update requests exactly one frame.
 * useGSAP kills every tween and ScrollTrigger created here when App unmounts.
 */
export function useScrollChoreography(scope) {
  useGSAP(
    () => {
      const reduce = prefersReducedMotion()
      const { hero, rules, gnn, llm, approach, inside, standards, install } = KEYS
      const camera = ['camY', 'camZ', 'orbit']
      const layers = gsap.utils.toArray('[data-layer]')

      const SEGMENT = { rules: 0, gnn: 1, llm: 2, sandbox: 3, exit: 4, install: 5 }
      const segments = [
        // Each intelligence layer finishes its scene exactly when its text is centered.
        { trigger: layers[0], start: 'top bottom', end: 'center center' },
        { trigger: layers[0], start: 'center center', endTrigger: layers[1], end: 'center center' },
        { trigger: layers[1], start: 'center center', endTrigger: layers[2], end: 'center center' },
        { trigger: layers[2], start: 'center center', endTrigger: '#sandbox', end: 'bottom bottom' },
        { trigger: '#standards', start: 'top bottom', end: 'top 20%' },
        { trigger: '#install', start: 'top bottom', end: 'bottom bottom' },
      ]

      const master = gsap.timeline({ paused: true, defaults: { ease: 'none' }, onUpdate: requestFrame })
      const at = (segment, tenth) => segment + tenth / 10
      const tween = (segment, from, to, start, duration, ease = 'sine.inOut') =>
        master.fromTo(
          choreo,
          { ...from },
          { ...to, duration: duration / 10, ease, immediateRender: false },
          at(segment, start),
        )
      const reveal = (segment, targets, from, to, start, duration) =>
        master.fromTo(targets, from, { ...to, duration: duration / 10, immediateRender: false }, at(segment, start))

      // Rule engine: the scan plane sweeps the tree and the smells light up.
      tween(SEGMENT.rules, pick(hero, camera), rules, 0, 10)
      tween(SEGMENT.rules, { scan: 0 }, { scan: 1 }, 1, 8, 'none')

      // GNN: the tree regroups into per-function clusters; messages pass along the edges.
      tween(SEGMENT.gnn, pick(rules, camera), gnn, 0, 10)
      tween(SEGMENT.gnn, { morph: 0 }, { morph: 1 }, 0, 9, 'none')
      tween(SEGMENT.gnn, { pulse: 0.35 }, { pulse: 1 }, 1, 5)

      // LLM: proposals flicker on the smelly nodes until the tests decide.
      tween(SEGMENT.llm, { orbit: gnn.orbit }, llm, 0, 10)
      tween(SEGMENT.llm, { propose: 0 }, { propose: 1 }, 1, 6)
      tween(SEGMENT.llm, { pulse: 1 }, { pulse: 0.5 }, 3, 6)

      // Sandbox: the code is pulled into the glass, the camera enters, the sandbox seals, tests run.
      const s = SEGMENT.sandbox
      tween(s, { collapse: 0 }, { collapse: 1 }, 0, 2.6, 'none')
      tween(s, { labels: 1 }, { labels: 0 }, 0, 1.2)
      tween(s, { pulse: 0.5 }, { pulse: 0 }, 0, 1.5)
      tween(s, { ...pick(gnn, ['camY', 'camZ']), orbit: llm.orbit, lens: hero.lens }, approach, 0, 2.6)
      tween(s, { glass: 0 }, { glass: 1 }, 0.9, 2.4, 'none')
      tween(s, { part: 0 }, { part: 1 }, 3.0, 1.2)
      tween(s, pick(approach, ['camY', 'camZ']), pick(inside, ['camY', 'camZ']), 3.4, 3.2, 'power2.inOut')
      tween(s, { lens: approach.lens }, { lens: inside.lens }, 3.4, 2.4)
      tween(s, { fov: hero.fov }, { fov: inside.fov }, 5.0, 1.6)
      tween(s, { part: 1 }, { part: 0 }, 6.7, 1.0)
      tween(s, { seal: 0 }, { seal: 1 }, 6.9, 1.0)
      tween(s, { verify: 0 }, { verify: 1 }, 7.9, 1.6, 'none')
      tween(s, { propose: 1 }, { propose: 0 }, 9.0, 0.6)
      tween(s, { heal: 0 }, { heal: 1 }, 9.0, 1.0)
      // The HTML layer is on the same timeline, so text and camera never drift apart.
      const protections = gsap.utils.toArray('[data-protection]')
      const gateResult = gsap.utils.toArray('[data-gate-result]')
      gsap.set(protections, { opacity: 0.3 })
      gsap.set(gateResult, { opacity: 0 })
      protections.forEach((el, i) => reveal(s, el, { opacity: 0.3 }, { opacity: 1 }, 3.0 + i * 1.25, 0.5))
      reveal(s, gateResult, { opacity: 0 }, { opacity: 1 }, 9.3, 0.6)

      // Standards: the sandbox opens and the code leaves as the clean, modernized tree.
      const e = SEGMENT.exit
      tween(e, { emerge: 0 }, { emerge: 1 }, 0, 0.3, 'none') // while still collapsed: an invisible switch
      tween(e, { part: 0 }, { part: 1 }, 0, 1.2)
      tween(e, { seal: 1 }, { seal: 0 }, 0, 1.0)
      tween(e, { fov: inside.fov }, { fov: standards.fov }, 0.6, 2.4)
      tween(e, { ...pick(inside, ['camY', 'camZ']), orbit: approach.orbit }, pick(standards, camera), 1.0, 5.4, 'power2.inOut')
      tween(e, { glass: 1 }, { glass: 0 }, 3.0, 3.0, 'none')
      tween(e, { collapse: 1 }, { collapse: 0 }, 3.2, 4.4, 'none')
      tween(e, { lens: inside.lens }, { lens: standards.lens }, 3.0, 4.0)
      // The modernized labels show briefly, then clear before the section's text arrives.
      tween(e, { labels: 0 }, { labels: 1 }, 6.0, 1.2)
      tween(e, { labels: 1 }, { labels: 0 }, 8.4, 1.4)
      tween(e, { pulse: 0 }, { pulse: 0.3 }, 7.0, 2)

      // Install: the modernized tree settles beside the closing call to action.
      tween(SEGMENT.install, pick(standards, [...camera, 'lens']), pick(install, [...camera, 'lens']), 0, 10)

      // Pad so the timeline is exactly one unit per segment.
      master.to({}, { duration: 0 }, segments.length)

      // Playhead = sum of segment progress, smoothed like GSAP's `scrub` but strictly ordered.
      const triggers = []
      const sync = () => {
        const time = triggers.reduce((sum, trigger) => sum + trigger.progress, 0)
        if (reduce) master.time(time)
        else gsap.to(master, { time, duration: 0.9, ease: 'power3.out', overwrite: true })
      }
      segments.forEach((segment) => triggers.push(ScrollTrigger.create({ ...segment, onUpdate: sync, onRefresh: sync })))

      // Load: the AST grows from its root, once.
      if (reduce) {
        choreo.intro = 1
        requestFrame()
      } else {
        gsap.to(choreo, { intro: 1, duration: 2.4, delay: 0.2, ease: 'power2.out', onUpdate: requestFrame })
      }

      // Web fonts change section heights; re-measure once they are in.
      document.fonts?.ready.then(() => ScrollTrigger.refresh())
    },
    { scope },
  )
}
