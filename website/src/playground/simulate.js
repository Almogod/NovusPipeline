/**
 * A small, faithful model of what the pipeline does with the README's example
 * `.novus/standards/payments.toml` on one legacy file. Rule semantics follow
 * standards.py: `add_keyword` and `replace_call` are automatic fixes (the new
 * import goes after the existing top-level imports), `imports` rules have no fix,
 * and `enforce = "block"` stops the commit while a violation remains.
 */

export const FILE_PATH = 'src/payments/charge.py'
export const TOML_PATH = '.novus/standards/payments.toml'

export const TOGGLES = [
  { id: 'timeout', label: 'Add timeout rule', hint: 'PAY-TIMEOUT adds timeout=10 to HTTP calls' },
  { id: 'pickle', label: 'Ban pickle', hint: 'PAY-NO-PICKLE has no automatic fix' },
  { id: 'block', label: 'Enforce block', hint: 'Refuse to commit while pickle remains', requires: 'pickle' },
  { id: 'ledger', label: 'Replace ledger client', hint: 'PAY-LEGACY-CLIENT renames the call' },
]

export const INITIAL_STATE = { timeout: true, pickle: true, block: false, ledger: true }

const SOURCE = [
  'import pickle',
  'import requests',
  'import ledger_v1',
  '',
  'GATEWAY = "https://pay.example.com/charges"',
  '',
  '',
  'def charge(customer, amount, token):',
  '    profile = pickle.loads(customer.blob)',
  '    resp = requests.post(GATEWAY, json={"token": token})',
  '    ledger_v1.post_entry(profile.account, amount)',
  '    return resp.json()',
]

const RULES = {
  timeout: {
    id: 'PAY-TIMEOUT',
    title: 'Outbound HTTP calls need a timeout',
    severity: 'MEDIUM',
    line: 10,
    fix: (code) => code.replace('json={"token": token})', 'json={"token": token}, timeout=10)'),
    fixNote: 'added timeout=10',
  },
  pickle: {
    id: 'PAY-NO-PICKLE',
    title: 'No pickle in the payment service',
    severity: 'HIGH',
    line: 1,
    fixNote: 'no automatic fix',
  },
  ledger: {
    id: 'PAY-LEGACY-CLIENT',
    title: 'Use the new ledger client',
    severity: 'MEDIUM',
    line: 11,
    fix: (code) => code.replace('ledger_v1.post_entry', 'ledger.post_entry'),
    fixNote: 'renamed to ledger.post_entry, import added',
  },
}

export function tomlBlocks(state) {
  const blocks = [
    {
      key: 'pack',
      lines: ['[pack]', 'name = "acme-payments"', 'description = "Payment service standards"'],
    },
  ]
  if (state.timeout)
    blocks.push({
      key: 'timeout',
      lines: [
        '[[rule]]',
        'id = "PAY-TIMEOUT"',
        'title = "Outbound HTTP calls need a timeout"',
        'severity = "MEDIUM"',
        'calls = ["requests.get", "requests.post"]',
        'require_keyword = "timeout"   # only calls without timeout=',
        'add_keyword = "timeout=10"    # automatic fix',
      ],
    })
  if (state.pickle)
    blocks.push({
      key: 'pickle',
      lines: [
        '[[rule]]',
        'id = "PAY-NO-PICKLE"',
        'title = "No pickle in the payment service"',
        'severity = "HIGH"',
        'imports = ["pickle"]',
        'paths = ["src/payments/**"]',
      ],
      optional: state.block ? { key: 'enforce', line: 'enforce = "block"            # refuse to commit while this remains' } : null,
    })
  if (state.ledger)
    blocks.push({
      key: 'ledger',
      lines: [
        '[[rule]]',
        'id = "PAY-LEGACY-CLIENT"',
        'title = "Use the new ledger client"',
        'calls = ["ledger_v1.post_entry"]',
        'replace_call = "ledger.post_entry"   # automatic fix',
      ],
    })
  return blocks
}

/** Runs the active rules over the sample file and returns findings, a line diff and the pipeline outcome. */
export function simulate(state) {
  const active = ['timeout', 'pickle', 'ledger'].filter((key) => state[key])
  const findings = active.map((key) => {
    const rule = RULES[key]
    const blocking = key === 'pickle' && state.block
    return { key, ...rule, fixed: Boolean(rule.fix), blocking }
  })

  // Build the diff line by line against the original file.
  const diff = []
  SOURCE.forEach((code, index) => {
    const line = index + 1
    const finding = findings.find((f) => f.line === line)
    let next = code
    for (const f of findings) if (f.fix && f.line === line) next = f.fix(next)
    if (next !== code) {
      diff.push({ kind: 'removed', code, line, rule: finding?.id })
      diff.push({ kind: 'added', code: next, rule: finding?.id })
    } else {
      diff.push({ kind: finding ? 'flagged' : 'same', code, line, rule: finding?.id, severity: finding?.severity })
    }
    // replace_call adds its import after the existing top-level imports.
    if (state.ledger && code === 'import ledger_v1') diff.push({ kind: 'added', code: 'import ledger', rule: 'PAY-LEGACY-CLIENT' })
  })

  const fixedCount = findings.filter((f) => f.fixed).length
  const blocking = findings.filter((f) => f.blocking)
  const warnings = findings.filter((f) => !f.fixed && !f.blocking)

  let outcome
  if (!findings.length) {
    outcome = {
      status: 'idle',
      title: 'No standards active',
      detail: 'Built-in rules and Ruff still run on every audit. Turn on a rule to see what it does.',
    }
  } else if (blocking.length) {
    outcome = {
      status: 'blocked',
      title: 'Pipeline stopped. Nothing was committed.',
      detail: `${FILE_PATH} still violates a blocking standard: L1 ${blocking[0].id} ${blocking[0].title}. It needs a manual change (or use_llm=true).`,
    }
  } else if (fixedCount) {
    outcome = {
      status: 'committed',
      title: `Tests passed. Committed ${commitHash(state)} on auto-modernization-branch.`,
      detail: warnings.length
        ? `${warnings.length} warning left for review: ${warnings.map((w) => w.id).join(', ')}.`
        : 'Your working copy and current branch are unchanged.',
    }
  } else {
    outcome = {
      status: 'warned',
      title: `Nothing to modernize automatically in ${FILE_PATH}.`,
      detail: `Reported for review: ${warnings.map((w) => w.id).join(', ')}.`,
    }
  }

  return { findings, diff, fixedCount, blockingCount: blocking.length, outcome }
}

function commitHash(state) {
  const hashes = ['3f2a9c1e', 'b71d04e9', '9c1e4b2a', 'e05c7f31', '6ad2903b', 'c48e1f57', '1b9f6a2d', '7e3c58a0']
  const index = (state.timeout ? 1 : 0) + (state.ledger ? 2 : 0) + (state.pickle ? 4 : 0)
  return hashes[index]
}
