import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { FILE_PATH, INITIAL_STATE, TOGGLES, TOML_PATH, simulate, tomlBlocks } from './simulate'

const reveal = {
  initial: { height: 0, opacity: 0 },
  animate: { height: 'auto', opacity: 1 },
  exit: { height: 0, opacity: 0 },
  transition: { duration: 0.32, ease: [0.2, 0.7, 0.2, 1] },
}

function Switch({ on, disabled, label, hint, onChange }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={onChange}
      className="flex items-start gap-3 rounded-lg p-1 text-left transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
    >
      <span
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full border transition-colors ${
          on ? 'border-glacier bg-glacier/85' : 'border-rule bg-hull-2'
        }`}
      >
        <motion.span
          className={`absolute top-[2px] left-[2px] size-3.5 rounded-full ${on ? 'bg-midnight' : 'bg-fog'}`}
          animate={{ x: on ? 16 : 0 }}
          transition={{ type: 'spring', stiffness: 500, damping: 34 }}
        />
      </span>
      <span>
        <span className="block text-sm font-medium text-paper">{label}</span>
        <span className="mt-0.5 block text-label text-fog">{hint}</span>
      </span>
    </button>
  )
}

function TomlLine({ text }) {
  if (text.startsWith('[')) return <span className="text-glacier">{text}</span>
  const match = text.match(/^(\w+)(\s*=\s*)(.*?)(\s+#.*)?$/)
  if (!match) return <span>{text}</span>
  const [, key, equals, value, comment] = match
  return (
    <>
      <span className="text-[#9fb3d1]">{key}</span>
      <span className="text-fog">{equals}</span>
      <span className="text-paper">{value}</span>
      {comment && <span className="text-fog/80 italic">{comment}</span>}
    </>
  )
}

function TomlView({ state }) {
  const blocks = tomlBlocks(state)
  return (
    <pre className="overflow-x-auto px-5 py-4 font-mono text-[0.78rem] leading-[1.7]">
      <code>
        <AnimatePresence initial={false}>
          {blocks.map((block, b) => (
            <motion.div key={block.key} {...reveal} className="overflow-hidden">
              <div className={b ? 'pt-[1.7em]' : ''}>
                {block.lines.map((line) => (
                  <div key={line}>
                    <TomlLine text={line} />
                  </div>
                ))}
                <AnimatePresence initial={false}>
                  {block.optional && (
                    <motion.div key={block.optional.key} {...reveal} className="-mx-5 overflow-hidden">
                      <div className="border-l-2 border-block bg-block/10 px-[calc(1.25rem-2px)]">
                        <TomlLine text={block.optional.line} />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </code>
    </pre>
  )
}

const LINE_STYLE = {
  same: { sign: ' ', row: '', text: 'text-paper/75' },
  flagged: { sign: '!', row: 'bg-sodium/[0.08]', text: 'text-paper', signColor: 'text-sodium' },
  removed: { sign: '−', row: 'bg-block/[0.08]', text: 'text-paper/45 line-through decoration-block/50', signColor: 'text-block' },
  added: { sign: '+', row: 'bg-glacier/[0.09]', text: 'text-paper', signColor: 'text-glacier' },
}

function DiffView({ diff }) {
  return (
    <pre className="overflow-x-auto py-3 font-mono text-[0.78rem] leading-[1.7]">
      <code className="block min-w-max">
        {diff.map((row, i) => {
          const style = LINE_STYLE[row.kind]
          return (
            <div key={`${row.kind}-${row.code}-${i}`} className={`flex pr-5 ${style.row}`}>
              <span className="w-10 shrink-0 pr-3 text-right text-fog/50 select-none">{row.line ?? ''}</span>
              <span className={`w-4 shrink-0 select-none ${style.signColor ?? ''}`} aria-hidden="true">
                {style.sign}
              </span>
              <span className={style.text}>{row.code || ' '}</span>
              {row.kind === 'flagged' && <span className="sr-only">Flagged by {row.rule}</span>}
              {row.kind === 'removed' && <span className="sr-only">Removed</span>}
              {row.kind === 'added' && <span className="sr-only">Added by {row.rule}</span>}
            </div>
          )
        })}
      </code>
    </pre>
  )
}

const OUTCOME_STYLE = {
  committed: 'border-glacier/35 bg-glacier/[0.07] text-glacier',
  blocked: 'border-block/40 bg-block/[0.08] text-block',
  warned: 'border-sodium/35 bg-sodium/[0.07] text-sodium',
  idle: 'border-rule bg-hull-2/60 text-fog',
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? '' : 's'}`
}

export function Playground() {
  const [state, setState] = useState(INITIAL_STATE)
  const result = useMemo(() => simulate(state), [state])

  // The panel's height can change; keep the scroll-driven sections below measured correctly.
  useEffect(() => {
    const id = setTimeout(() => ScrollTrigger.refresh(), 400)
    return () => clearTimeout(id)
  }, [state])

  const toggle = (id) =>
    setState((s) => {
      const next = { ...s, [id]: !s[id] }
      if (id === 'pickle' && !next.pickle) next.block = false
      return next
    })

  const { findings, fixedCount, blockingCount, outcome } = result

  return (
    <div className="overflow-hidden rounded-2xl border border-rule bg-hull/90 shadow-[0_50px_140px_-60px_rgb(139_211_255/0.25)] backdrop-blur-xl">
      <div
        role="group"
        aria-label={`Rules in ${TOML_PATH}`}
        className="grid gap-x-6 gap-y-3 border-b border-rule p-4 sm:grid-cols-2 md:p-5 xl:grid-cols-4"
      >
        {TOGGLES.map((t) => (
          <Switch
            key={t.id}
            on={state[t.id]}
            disabled={t.requires && !state[t.requires]}
            label={t.label}
            hint={t.hint}
            onChange={() => toggle(t.id)}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
        <div className="border-b border-rule lg:border-r lg:border-b-0">
          <div className="flex h-11 items-center border-b border-rule px-5 font-mono text-[0.75rem] text-fog">
            {TOML_PATH}
          </div>
          <div className="lg:min-h-[34rem]">
            <TomlView state={state} />
          </div>
        </div>

        <div className="flex min-w-0 flex-col">
          <div className="flex h-11 items-center justify-between gap-4 border-b border-rule px-5 font-mono text-[0.75rem] text-fog">
            <span className="truncate">{FILE_PATH}</span>
            <span className="shrink-0 font-sans">Simulated run</span>
          </div>
          <DiffView diff={result.diff} />

          <div className="mt-auto space-y-3 border-t border-rule p-4 md:p-5">
            <p className="text-sm text-fog" aria-live="polite">
              {findings.length
                ? `${plural(findings.length, 'finding')} from your standards: ${fixedCount} fixed automatically${
                    blockingCount ? `, ${blockingCount} blocking` : ''
                  }.`
                : 'No findings from your standards.'}
            </p>
            <ul className="space-y-1.5">
              <AnimatePresence initial={false}>
                {findings.map((f) => (
                  <motion.li key={f.key} {...reveal} className="overflow-hidden">
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-sm">
                      <span className={`font-mono text-[0.78rem] ${f.blocking ? 'text-block' : f.fixed ? 'text-glacier' : 'text-sodium'}`}>
                        L{f.line} {f.id}
                      </span>
                      <span className="text-paper/85">{f.title}</span>
                      <span className="text-label text-fog">
                        {f.severity.toLowerCase()}, {f.blocking ? 'blocks the commit' : f.fixNote}
                      </span>
                    </div>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={outcome.status + outcome.title}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
                className={`rounded-lg border px-4 py-3 ${OUTCOME_STYLE[outcome.status]}`}
                role="status"
              >
                <p className="text-sm font-semibold">{outcome.title}</p>
                <p className="mt-1 text-label text-paper/70">{outcome.detail}</p>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}
