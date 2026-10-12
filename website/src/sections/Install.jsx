import { useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CopyCommand } from '../components/CopyCommand'
import { LogoMark } from '../components/Logo'
import { CHANGELOG_URL, CLIENT_SETUPS, DOCS_URL, GITHUB_URL, PYPI_URL, SECURITY_URL, UVX_COMMAND } from '../lib/site'

function ClientTabs() {
  const [active, setActive] = useState(CLIENT_SETUPS[0].id)
  const tabs = useRef([])
  const current = CLIENT_SETUPS.find((s) => s.id === active)

  function onKeyDown(event, index) {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key]
    if (!step) return
    const next = (index + step + CLIENT_SETUPS.length) % CLIENT_SETUPS.length
    setActive(CLIENT_SETUPS[next].id)
    tabs.current[next]?.focus()
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-rule bg-hull/85 backdrop-blur-xl">
      <div role="tablist" aria-label="MCP client" className="flex gap-1 overflow-x-auto border-b border-rule px-2">
        {CLIENT_SETUPS.map((setup, i) => (
          <button
            key={setup.id}
            ref={(el) => (tabs.current[i] = el)}
            type="button"
            role="tab"
            id={`tab-${setup.id}`}
            aria-selected={active === setup.id}
            aria-controls="client-panel"
            tabIndex={active === setup.id ? 0 : -1}
            onClick={() => setActive(setup.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={`relative shrink-0 px-3 py-3 text-sm transition-colors ${
              active === setup.id ? 'text-paper' : 'text-fog hover:text-paper'
            }`}
          >
            {setup.label}
            {active === setup.id && (
              <motion.span layoutId="client-tab" className="absolute inset-x-3 -bottom-px h-px bg-glacier" />
            )}
          </button>
        ))}
      </div>
      <div id="client-panel" role="tabpanel" aria-labelledby={`tab-${active}`} tabIndex={0}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.pre
            key={active}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-x-auto p-5 font-mono text-[0.8rem] leading-[1.7] text-paper/90"
          >
            <code>{current.code}</code>
          </motion.pre>
        </AnimatePresence>
      </div>
    </div>
  )
}

const FOOTER_LINKS = [
  { href: GITHUB_URL, label: 'GitHub' },
  { href: PYPI_URL, label: 'PyPI' },
  { href: DOCS_URL, label: 'Documentation' },
  { href: CHANGELOG_URL, label: 'Changelog' },
  { href: SECURITY_URL, label: 'Security' },
]

export function Install() {
  return (
    <section id="install" className="relative overflow-hidden pt-[20vh]" aria-labelledby="install-title">
      <div className="mx-auto max-w-[1240px] px-5 md:px-10">
        <div className="scrim max-w-[40rem]">
          <h2 id="install-title" className="type-h2 text-paper">
            Run it on your codebase
          </h2>
          <p className="mt-5 text-lead text-paper/80">
            One command starts the server. Add it to your MCP client, then ask your agent to use the{' '}
            <code className="font-mono text-[0.92em]">modernization_workflow</code> prompt.
          </p>
        </div>

        {/* The one place both accents meet: legacy (sodium) flowing into modern (glacier). */}
        <div className="relative mt-14 md:mt-20">
          <div
            aria-hidden="true"
            className="absolute -inset-x-10 -inset-y-8 bg-[radial-gradient(40%_70%_at_20%_50%,rgb(255_165_61/0.22),transparent),radial-gradient(45%_80%_at_70%_50%,rgb(139_211_255/0.24),transparent)] blur-2xl"
          />
          <p className="relative bg-gradient-to-r from-sodium via-paper to-glacier bg-clip-text font-mono text-[clamp(1.9rem,6.2vw,5.25rem)] leading-none font-medium tracking-[-0.04em] text-transparent">
            {UVX_COMMAND}
          </p>
        </div>

        <div className="mt-10 flex flex-wrap items-center gap-x-6 gap-y-3">
          <CopyCommand command={UVX_COMMAND} />
          <p className="text-sm text-fog">Requires Python 3.11+ and uv. Prefer pip? See the tabs below.</p>
        </div>

        <div className="mt-12 max-w-[56rem]">
          <ClientTabs />
          <p className="mt-4 text-sm text-fog">
            The <code className="font-mono text-paper/80">[gnn]</code> extra adds the structural GNN;{' '}
            <code className="font-mono text-paper/80">[llm]</code> adds local LLM proposals.
          </p>
        </div>

        <footer className="mt-[18vh] flex flex-col gap-6 border-t border-rule py-10 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2.5 text-paper">
            <LogoMark className="size-5" />
            <span className="text-sm font-semibold [font-stretch:104%]">NovusPipeline</span>
            <span className="text-sm text-fog">MIT licensed</span>
          </div>
          <ul className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-fog">
            {FOOTER_LINKS.map((link) => (
              <li key={link.label}>
                <a href={link.href} className="rounded-sm transition-colors hover:text-paper">
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </footer>
      </div>
    </section>
  )
}
