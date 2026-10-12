import { Playground } from '../playground/Playground'

export function Standards() {
  return (
    <section id="standards" className="relative py-[18vh]" aria-labelledby="standards-title">
      <div className="mx-auto max-w-[1240px] px-5 md:px-10">
        <div className="scrim max-w-[38rem]">
          <h2 id="standards-title" className="type-h2 text-paper">
            Your standards, enforced
          </h2>
          <p className="mt-5 text-lead text-paper/80">
            Write team or industry rules as TOML in <code className="font-mono text-[0.92em]">.novus/standards/</code>.
            Audits report them, the pipeline applies their fixes, LLM proposals must follow them, and a rule with{' '}
            <code className="font-mono text-[0.92em]">enforce = "block"</code> stops the commit. Try it on a payment
            service:
          </p>
        </div>

        <div className="mt-12">
          <Playground />
        </div>

        <p className="mt-6 max-w-[44rem] text-sm text-fog">
          Or start from a built-in pack with <code className="font-mono text-paper/80">install_standard_pack</code>:{' '}
          <code className="font-mono text-paper/80">secure-defaults</code> covers HTTP timeouts, safe YAML, TLS
          verification and <code className="font-mono text-paper/80">shell=True</code>;{' '}
          <code className="font-mono text-paper/80">python-312-ready</code> covers APIs removed in Python 3.12.
        </p>
      </div>
    </section>
  )
}
