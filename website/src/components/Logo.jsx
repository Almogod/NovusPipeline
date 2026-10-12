/** A three-node AST: the root, one legacy child (sodium) and one modernized child (glacier). */
export function LogoMark({ className = 'size-6' }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        d="M12 5 6.5 18M12 5l5.5 13"
        stroke="currentColor"
        strokeOpacity=".5"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="12" cy="5" r="2.75" fill="currentColor" />
      <circle cx="6.5" cy="18" r="2.75" fill="var(--color-sodium)" />
      <circle cx="17.5" cy="18" r="2.75" fill="var(--color-glacier)" />
    </svg>
  )
}
