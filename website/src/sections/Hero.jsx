import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { CopyCommand } from '../components/CopyCommand'
import { GitHubIcon } from '../components/Icons'
import { GITHUB_URL, UVX_COMMAND } from '../lib/site'

const LINES = [['Modernize', 'Legacy'], ['Codebases.'], ['Fast.', 'Local.', 'Safe.']]

/** Waits for the display face so the headline animation never runs on a fallback font. */
function useFontReady(descriptor) {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    let alive = true
    const done = () => alive && setReady(true)
    const timeout = setTimeout(done, 1500)
    document.fonts?.load(descriptor).then(done, done) ?? done()
    return () => {
      alive = false
      clearTimeout(timeout)
    }
  }, [descriptor])
  return ready
}

const fade = (delay) => ({
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { delay, duration: 0.9, ease: [0.2, 0.7, 0.2, 1] },
})

export function Hero() {
  const ready = useFontReady('760 64px "Archivo Variable"')
  let wordIndex = 0

  return (
    <section id="top" className="relative flex min-h-svh items-center">
      <div className="mx-auto w-full max-w-[1240px] px-5 pt-28 pb-24 md:px-10">
        <div className="scrim max-w-[37rem]">
          <h1 className="headline text-paper" data-ready={ready}>
            {LINES.map((line, l) => (
              <span key={l} className="block sm:whitespace-nowrap">
                {line.map((word, w) => (
                  <span key={word}>
                    <span className="word" style={{ '--i': wordIndex++ }}>
                      {word}
                    </span>
                    {w < line.length - 1 && ' '}
                  </span>
                ))}
              </span>
            ))}
          </h1>

          <motion.p {...fade(0.9)} className="mt-8 max-w-[33rem] text-lead text-paper/80">
            NovusPipeline is an MCP server for your coding agent. It finds legacy smells with rules, Ruff and a graph
            neural network that reads your AST, proposes fixes with an optional local LLM, and commits only what passes
            your tests in a sandbox. Nothing leaves your machine.
          </motion.p>

          <motion.div {...fade(1.15)} className="mt-9 flex flex-wrap items-center gap-3">
            <CopyCommand command={UVX_COMMAND} />
            <a
              href={GITHUB_URL}
              className="flex h-[3.25rem] items-center gap-2.5 rounded-xl px-4 text-paper transition-colors hover:bg-hull/80"
            >
              <GitHubIcon className="size-5" />
              GitHub repo
            </a>
          </motion.div>

          <motion.p {...fade(1.4)} className="mt-6 text-label text-fog">
            Works with Claude Code, Claude Desktop, Cursor and other MCP clients. Python 3.11+, MIT licensed.
          </motion.p>
        </div>
      </div>
    </section>
  )
}
