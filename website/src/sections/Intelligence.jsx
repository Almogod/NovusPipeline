import { useRef, useState } from 'react'
import { motion } from 'motion/react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useGSAP } from '@gsap/react'

gsap.registerPlugin(ScrollTrigger, useGSAP)

// Each layer is an install tier: the label is what you install to get it.
const LAYERS = [
  {
    install: 'uvx novuspipeline',
    title: 'Rule engine',
    body: "Built-in legacy rules, Ruff, your team's rule packs, codebase-wide checks for import cycles, dead code and clones, and smells learned from your own git history. Rewrites are behavior-preserving, and any result that no longer parses is thrown away.",
    fact: 'Runs anywhere Python 3.11 does. No GPU, no model download.',
  },
  {
    install: 'novuspipeline[gnn]',
    title: 'Structural GNN',
    body: 'A GraphSAGE network reads each file as a graph of AST nodes, fields and identifiers. It cross-checks the rules, points to the exact functions that smell, and finds structurally similar code across the repository.',
    fact: '0.998 file-level macro-F1 on 19 packages it never saw in training. The 1.2 MB checkpoint ships with the package and runs on the CPU.',
  },
  {
    install: 'novuspipeline[llm]',
    title: 'Local LLM proposals',
    body: "Qwen3.5-2B runs on your own GPU and proposes rewrites the rules can't make. A proposal is kept only if it parses, adds no new smells or standard violations, and keeps every symbol other modules import. Then your tests decide.",
    fact: 'Opt in with use_llm. About 4 GB of free VRAM recommended. Falls back to the rule engine.',
  },
]

export function Intelligence() {
  const scope = useRef(null)
  const [active, setActive] = useState(0)

  // Which layer is centered decides which one is lit; the 3D scene follows the same scroll.
  useGSAP(
    () => {
      gsap.utils.toArray('[data-layer]').forEach((el, i) =>
        ScrollTrigger.create({
          trigger: el,
          start: 'top 55%',
          end: 'bottom 45%',
          onToggle: (self) => self.isActive && setActive(i),
        }),
      )
    },
    { scope },
  )

  return (
    <section id="intelligence" ref={scope} className="relative" aria-labelledby="intelligence-title">
      <div className="mx-auto max-w-[1240px] px-5 md:px-10">
        <header className="scrim max-w-[32rem] pt-[22vh]">
          <h2 id="intelligence-title" className="type-h2 text-paper">
            Three layers of local intelligence
          </h2>
          <p className="mt-5 text-lead text-paper/80">
            Start with the rule engine. Add the structural GNN or a local LLM when you want them. Every layer runs on
            your machine, and every change still has to pass your tests.
          </p>
        </header>

        <div className="mt-[6vh]">
          {LAYERS.map((layer, i) => (
            <article key={layer.title} data-layer className="flex min-h-[78vh] items-center">
              <div
                className={`scrim relative max-w-[30rem] pl-6 transition-opacity duration-500 md:pl-8 ${
                  active === i ? 'opacity-100' : 'opacity-40'
                }`}
              >
                <span className="absolute top-0 left-0 h-full w-px bg-rule" aria-hidden="true" />
                {active === i && (
                  <motion.span
                    layoutId="layer-indicator"
                    className="absolute top-0 left-0 h-full w-px bg-glacier shadow-[0_0_14px_1px_var(--color-glacier)]"
                    transition={{ type: 'spring', stiffness: 260, damping: 32 }}
                    aria-hidden="true"
                  />
                )}
                <p className="font-mono text-label text-glacier">{layer.install}</p>
                <h3 className="mt-3 text-h3 font-semibold text-paper [font-stretch:104%]">{layer.title}</h3>
                <p className="mt-4 text-paper/80">{layer.body}</p>
                <p className="mt-5 text-label text-fog">{layer.fact}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
