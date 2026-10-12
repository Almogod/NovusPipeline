import { useEffect, useState } from 'react'
import { LogoMark } from '../components/Logo'
import { GitHubIcon } from '../components/Icons'
import { GITHUB_URL } from '../lib/site'

const LINKS = [
  { href: '#intelligence', label: 'How it works' },
  { href: '#sandbox', label: 'Sandbox' },
  { href: '#standards', label: 'Standards' },
  { href: '#install', label: 'Install' },
]

export function Nav() {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const update = () => setScrolled(window.scrollY > 24)
    update()
    window.addEventListener('scroll', update, { passive: true })
    return () => window.removeEventListener('scroll', update)
  }, [])

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-[background-color,border-color,backdrop-filter] duration-300 ${
        scrolled ? 'border-b border-rule/70 bg-midnight/70 backdrop-blur-xl' : 'border-b border-transparent'
      }`}
    >
      <nav aria-label="Main" className="mx-auto flex h-16 max-w-[1240px] items-center justify-between px-5 md:px-10">
        <a href="#top" className="flex items-center gap-2.5 rounded-md text-paper">
          <LogoMark />
          <span className="text-[1.05rem] font-semibold tracking-[-0.01em] [font-stretch:104%]">NovusPipeline</span>
        </a>
        <ul className="hidden items-center gap-8 text-sm text-fog md:flex">
          {LINKS.map((link) => (
            <li key={link.href}>
              <a href={link.href} className="rounded-sm transition-colors hover:text-paper">
                {link.label}
              </a>
            </li>
          ))}
        </ul>
        <a
          href={GITHUB_URL}
          className="flex items-center gap-2 rounded-lg border border-rule px-3 py-1.5 text-sm text-paper transition-colors hover:border-fog/60 hover:bg-hull"
        >
          <GitHubIcon className="size-4" />
          GitHub
        </a>
      </nav>
    </header>
  )
}
