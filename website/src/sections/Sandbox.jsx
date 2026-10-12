import { LockIcon } from '../components/Icons'

// From the README's sandbox protection table; the network row is stated with its real scope.
const PROTECTIONS = [
  {
    title: 'Disposable worktree',
    body: "The file is modernized in a temporary git worktree and committed to a branch you don't have checked out. Your working copy is never touched.",
  },
  {
    title: 'Scrubbed environment',
    body: "Tokens and API keys are hidden from tests. Home and temp point into the sandbox, so ~/.ssh and ~/.aws aren't found through them.",
  },
  {
    title: 'Hard limits',
    body: 'Memory and process caps, and the whole process tree is killed on timeout.',
  },
  {
    title: 'No network',
    body: 'Built in on Linux. On Windows and macOS, set NOVUS_SANDBOX=docker to run tests in a container with --network none.',
  },
]

/**
 * Tall section with a sticky panel: while it is pinned, the camera flies into the
 * glass sandbox and the protections light up in step (see useScrollChoreography).
 */
export function Sandbox() {
  return (
    <section id="sandbox" className="relative h-[420vh]" aria-labelledby="sandbox-title">
      <div className="sticky top-0 flex h-svh items-end pb-8 md:items-center md:pb-0">
        <div className="mx-auto w-full max-w-[1240px] px-5 md:px-10">
          <div className="max-w-[27rem] rounded-2xl border border-rule/70 bg-midnight/70 p-6 backdrop-blur-xl md:p-7">
            <h2 id="sandbox-title" className="type-h2 text-paper">
              Test-gated sandbox
            </h2>
            <p className="mt-4 text-lead text-paper/80">
              Every change is tested before it is committed, and never in your working tree.
            </p>
            <ul className="mt-7 space-y-4 md:mt-8 md:space-y-5">
              {PROTECTIONS.map((item) => (
                <li key={item.title} data-protection className="flex gap-3.5">
                  <LockIcon className="mt-0.5 size-5 shrink-0 text-glacier" />
                  <div>
                    <p className="font-medium text-paper">{item.title}</p>
                    <p className="mt-0.5 text-sm text-fog max-md:hidden">{item.body}</p>
                  </div>
                </li>
              ))}
            </ul>
            <p
              data-gate-result
              className="mt-7 rounded-lg border border-glacier/25 bg-hull/70 px-4 py-3 font-mono text-[0.8rem] leading-relaxed text-paper/85 backdrop-blur-md"
            >
              <span className="text-glacier">Tests passed (exit code 0).</span> Committed 9c1e4b2a on
              auto-modernization-branch; your working copy and current branch are unchanged.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
