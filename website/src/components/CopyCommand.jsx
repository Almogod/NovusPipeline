import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CheckIcon, CopyIcon } from './Icons'

async function writeClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Older browsers and insecure origins: fall back to a hidden textarea.
    const area = Object.assign(document.createElement('textarea'), { value: text })
    area.setAttribute('readonly', '')
    area.style.cssText = 'position:fixed;opacity:0'
    document.body.append(area)
    area.select()
    const ok = document.execCommand('copy')
    area.remove()
    return ok
  }
}

/** A command shown as code; clicking copies it. The label confirms what happened: "Copy" -> "Copied". */
export function CopyCommand({ command, className = '', prompt = true }) {
  const [state, setState] = useState('idle') // idle | copied | failed
  const timer = useRef(0)
  useEffect(() => () => clearTimeout(timer.current), [])

  async function copy() {
    const ok = await writeClipboard(command)
    setState(ok ? 'copied' : 'failed')
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setState('idle'), 1800)
  }

  const label = { idle: 'Copy', copied: 'Copied', failed: 'Select and copy' }[state]

  return (
    <div className={`flex items-stretch ${className}`}>
      <button
        type="button"
        onClick={copy}
        className="group flex min-w-0 items-center gap-4 rounded-xl border border-glacier/30 bg-hull/80 py-3 pr-3 pl-4 text-left backdrop-blur-md transition-colors hover:border-glacier/60 hover:bg-hull"
        aria-label={`Copy command: ${command}`}
      >
        <code className="min-w-0 truncate font-mono text-[0.95rem] text-paper select-all">
          {prompt && <span className="mr-2 text-fog select-none">$</span>}
          {command}
        </code>
        <span className="relative flex h-8 shrink-0 items-center gap-1.5 rounded-lg bg-glacier/10 px-2.5 text-label font-medium text-glacier transition-colors group-hover:bg-glacier/20">
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={state}
              className="flex items-center gap-1.5"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.16 }}
            >
              {state === 'copied' ? <CheckIcon className="size-4" /> : <CopyIcon className="size-4" />}
              {label}
            </motion.span>
          </AnimatePresence>
        </span>
      </button>
      <span className="sr-only" role="status" aria-live="polite">
        {state === 'copied' ? `Copied ${command}` : ''}
      </span>
    </div>
  )
}
