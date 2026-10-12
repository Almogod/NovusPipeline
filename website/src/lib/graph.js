/**
 * A procedural Python AST and the layouts the scene morphs between.
 *
 * The tree is a plausible legacy module (`billing.py`). Its smell nodes are the
 * rule engine's real findings, and each label's "modern" text is the rewrite the
 * rule engine (or the secure-defaults pack) actually applies.
 *
 * Layouts, all Float32Array(n * 3):
 *   legacy  cone tree with jitter (hero)
 *   modern  the same cone tree, clean: same structure, behavior preserved
 *   cluster nodes regrouped per top-level definition, the way the GNN localizes smells
 *   core    a tight ball inside the sandbox
 */

export const LABELS = {
  root: { legacy: 'billing.py' },
  urllib2: { legacy: 'import urllib2', modern: 'import urllib.request', tag: 'PY-SMELL-002' },
  popen: { legacy: 'os.popen(cmd).read()', modern: 'subprocess.run(cmd, …).stdout', tag: 'PY-SMELL-004' },
  except: { legacy: 'except:', modern: 'except Exception:', tag: 'PY-SMELL-003' },
  print: { legacy: 'print "total", t', modern: 'print("total", t)', tag: 'PY-SMELL-001' },
  timeout: { legacy: 'requests.post(url, json=body)', modern: 'requests.post(…, timeout=30)', tag: 'SEC-TIMEOUT' },
}

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function node(type, kids = [], extra = {}) {
  return { type, kids, ...extra }
}

function buildAst(rand, detail) {
  const pick = (list) => list[Math.floor(rand() * list.length)]
  const count = (k) => Math.max(1, Math.round(k * detail))

  function expr(depth) {
    const type = depth <= 0 ? pick(['Name', 'Constant']) : pick(['Call', 'Attribute', 'Name', 'BinOp', 'Compare', 'Subscript'])
    if (type === 'Name' || type === 'Constant') return node(type)
    const arity = type === 'Call' ? 1 + Math.floor(rand() * 3) : 2
    return node(type, Array.from({ length: arity }, () => expr(depth - 1 - (rand() < 0.45 ? 1 : 0))))
  }

  function stmt(depth) {
    const type = pick(['Assign', 'Assign', 'Expr', 'Return', 'If', 'For'])
    if ((type === 'If' || type === 'For') && depth > 0) {
      return node(type, [expr(1), ...Array.from({ length: 1 + Math.floor(rand() * 2) }, () => stmt(depth - 1))])
    }
    if (type === 'Expr' || type === 'Return') return node(type, [expr(2)])
    return node('Assign', [node('Name'), expr(2)])
  }

  const fn = (name, statements, special = [], { hinted = true } = {}) =>
    node(
      'FunctionDef',
      [
        // Missing type hints (PY-SMELL-006) are smells too, just unlabeled.
        node('arguments', [node('arg'), node('arg')], hinted ? {} : { smell: true }),
        ...special,
        ...Array.from({ length: count(statements) }, () => stmt(1)),
      ],
      { name },
    )

  return node(
    'Module',
    [
      node('Import', [node('alias')], { label: 'urllib2', smell: true }),
      node('Import', [node('alias')]),
      node('Import', [node('alias')]),
      fn('fetch_rates', 3, [node('Try', [stmt(1), node('ExceptHandler', [stmt(0)], { label: 'except', smell: true })])]),
      fn('export_ledger', 2, [
        node('Assign', [
          node('Name'),
          node('Call', [node('Attribute', [node('Call', [node('Attribute', [node('Name')]), node('Name')])])], {
            label: 'popen',
            smell: true,
          }),
        ]),
      ]),
      fn('report', 2, [node('Print', [node('Constant'), node('Name')], { label: 'print', smell: true })], {
        hinted: false,
      }),
      fn('notify', 2, [
        node('Expr', [
          node('Call', [node('Attribute', [node('Name')]), node('Name'), node('keyword', [node('Name')])], {
            label: 'timeout',
            smell: true,
          }),
        ]),
      ]),
      node('ClassDef', [fn('__init__', 2), fn('total', 2, [], { hinted: false }), fn('render', 2)], { name: 'Invoice' }),
      fn('main', 3),
    ],
    { label: 'root' },
  )
}

/** Point `i` of `n` on a unit Fibonacci sphere. Consecutive indices stay close together. */
function fibonacci(i, n, out) {
  const y = 1 - (2 * (i + 0.5)) / n
  const r = Math.sqrt(Math.max(0, 1 - y * y))
  const theta = Math.PI * (3 - Math.sqrt(5)) * i
  out[0] = Math.cos(theta) * r
  out[1] = y
  out[2] = Math.sin(theta) * r
  return out
}

export function buildGraph({ detail = 1, seed = 7 } = {}) {
  const rand = mulberry32(seed)
  const ast = buildAst(rand, detail)

  // Flatten in preorder: parents come before children, subtrees are contiguous.
  const nodes = []
  const walk = (src, parent, depth, cluster, smell) => {
    const index = nodes.length
    const isSmell = smell || Boolean(src.smell)
    const entry = { type: src.type, parent, depth, cluster, smell: isSmell, label: src.label, kids: [] }
    nodes.push(entry)
    src.kids.forEach((kid, k) => {
      entry.kids.push(walk(kid, index, depth + 1, depth === 0 ? k : cluster, isSmell))
    })
    return index
  }
  walk(ast, -1, 0, -1, false)

  const n = nodes.length
  const clusterCount = ast.kids.length
  const maxDepth = nodes.reduce((m, nd) => Math.max(m, nd.depth), 0)

  // ---- Cone tree (Robertson et al.): children on a circle below their parent. ----
  const footprint = new Float32Array(n)
  const ring = new Float32Array(n)
  for (let i = n - 1; i >= 0; i--) {
    const kids = nodes[i].kids
    if (!kids.length) {
      footprint[i] = 0.16
      continue
    }
    let sum = 0
    let max = 0
    for (const c of kids) {
      sum += 2 * footprint[c]
      max = Math.max(max, footprint[c])
    }
    const r = kids.length === 1 ? 0.24 : Math.max(max * 1.02, (sum + 0.05 * kids.length) / (2 * Math.PI))
    // Shallow rings are compressed so the root cone does not dwarf the deep subtrees.
    ring[i] = r * (nodes[i].depth === 0 ? 0.62 : nodes[i].depth === 1 ? 0.8 : 1)
    footprint[i] = ring[i] + max
  }

  const cone = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const kids = nodes[i].kids
    if (!kids.length) continue
    const total = kids.reduce((s, c) => s + footprint[c], 0)
    const start = rand() * Math.PI * 2
    let acc = 0
    for (const c of kids) {
      const theta = start + (Math.PI * 2 * (acc + footprint[c] / 2)) / total
      acc += footprint[c]
      const gap = 1.0 * Math.pow(0.84, nodes[c].depth - 1)
      cone[c * 3] = cone[i * 3] + Math.cos(theta) * ring[i]
      cone[c * 3 + 1] = cone[i * 3 + 1] - gap
      cone[c * 3 + 2] = cone[i * 3 + 2] + Math.sin(theta) * ring[i]
    }
  }

  // Normalize: fit the cone into a ~5.8 wide, ~4.7 tall volume centered on the origin.
  let radius = 0
  let minY = Infinity
  let maxY = -Infinity
  for (let i = 0; i < n; i++) {
    radius = Math.max(radius, Math.hypot(cone[i * 3], cone[i * 3 + 2]))
    minY = Math.min(minY, cone[i * 3 + 1])
    maxY = Math.max(maxY, cone[i * 3 + 1])
  }
  const sxz = 2.9 / radius
  const sy = 4.7 / (maxY - minY)
  const midY = (maxY + minY) / 2
  const modern = new Float32Array(n * 3)
  const legacy = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    modern[i * 3] = cone[i * 3] * sxz
    modern[i * 3 + 1] = (cone[i * 3 + 1] - midY) * sy
    modern[i * 3 + 2] = cone[i * 3 + 2] * sxz
    // Legacy code is the same structure, just untidy.
    const j = nodes[i].depth === 0 ? 0 : 0.16
    legacy[i * 3] = modern[i * 3] + (rand() - 0.5) * 2 * j
    legacy[i * 3 + 1] = modern[i * 3 + 1] + (rand() - 0.5) * 2 * j * 0.8
    legacy[i * 3 + 2] = modern[i * 3 + 2] + (rand() - 0.5) * 2 * j
  }

  // ---- Clusters: one per top-level definition, around the Module hub. ----
  const cluster = new Float32Array(n * 3)
  const members = Array.from({ length: clusterCount }, () => [])
  nodes.forEach((nd, i) => nd.cluster >= 0 && members[nd.cluster].push(i))
  const dir = [0, 0, 0]
  members.forEach((list, k) => {
    fibonacci(k, clusterCount, dir)
    const cx = dir[0] * 2.6
    const cy = dir[1] * 1.7
    const cz = dir[2] * 2.6
    const scale = 0.45 + 0.4 * Math.cbrt(list.length / 30)
    list.forEach((i, j) => {
      const d = nodes[i].depth - 1
      const r = d === 0 ? 0 : (0.2 + 0.3 * Math.pow(d, 0.8)) * scale
      fibonacci(j, list.length, dir)
      cluster[i * 3] = cx + dir[0] * r + (rand() - 0.5) * 0.12
      cluster[i * 3 + 1] = cy + dir[1] * r + (rand() - 0.5) * 0.12
      cluster[i * 3 + 2] = cz + dir[2] * r + (rand() - 0.5) * 0.12
    })
  })

  // ---- Core: a small ball at the center of the sandbox. ----
  const core = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    fibonacci(i, n, dir)
    const r = 0.11 * Math.cbrt(rand())
    core[i * 3] = dir[0] * r
    core[i * 3 + 1] = dir[1] * r
    core[i * 3 + 2] = dir[2] * r
  }

  // ---- Per-node attributes. ----
  const scatter = new Float32Array(n * 3)
  const depth = new Float32Array(n)
  const size = new Float32Array(n)
  const tone = new Float32Array(n)
  const smell = new Uint8Array(n)
  const morphDelay = new Float32Array(n)
  const collapseDelay = new Float32Array(n)
  const phase = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const nd = nodes[i]
    depth[i] = nd.depth
    smell[i] = nd.smell ? 1 : 0
    const base = nd.depth === 0 ? 0.15 : nd.depth === 1 ? 0.105 : nd.depth === 2 ? 0.08 : 0.062
    size[i] = nd.label ? Math.max(base, 0.1) : nd.kids.length ? base : base * 0.88
    tone[i] = Math.min(1, nd.depth / 4) * (0.6 + 0.4 * rand())
    fibonacci(Math.floor(rand() * 997), 997, dir)
    const s = 0.5 + rand() * 0.7
    scatter[i * 3] = dir[0] * s
    scatter[i * 3 + 1] = dir[1] * s
    scatter[i * 3 + 2] = dir[2] * s
    morphDelay[i] = nd.cluster < 0 ? 0 : 0.38 * (nd.cluster / Math.max(1, clusterCount - 1)) + rand() * 0.04
    collapseDelay[i] = rand() * 0.3 + (nd.depth / maxDepth) * 0.12
    phase[i] = rand() * Math.PI * 2
  }

  // Edges: parent-child (the tree) and next-sibling (the GNN's extra edges, as in code_graph.py).
  const edgeList = []
  const edgeKind = []
  nodes.forEach((nd, i) => {
    nd.kids.forEach((c, k) => {
      edgeList.push(i, c)
      edgeKind.push(0)
      if (k > 0) {
        edgeList.push(nd.kids[k - 1], c)
        edgeKind.push(1)
      }
    })
  })
  const edges = Uint16Array.from(edgeList)
  const edgeType = Uint8Array.from(edgeKind)

  // Pulses travel parent -> child along a sample of tree edges.
  const treeEdges = []
  edgeType.forEach((t, e) => t === 0 && treeEdges.push(e))
  const pulseCount = Math.min(64, treeEdges.length)
  const pulses = Array.from({ length: pulseCount }, () => ({
    edge: treeEdges[Math.floor(rand() * treeEdges.length)],
    speed: 0.18 + rand() * 0.3,
    phase: rand(),
  }))

  const labels = nodes.flatMap((nd, i) => (nd.label ? [{ index: i, key: nd.label, ...LABELS[nd.label] }] : []))

  return {
    n,
    maxDepth,
    layouts: { legacy, modern, cluster, core },
    scatter,
    depth,
    size,
    tone,
    smell,
    morphDelay,
    collapseDelay,
    phase,
    edges,
    edgeType,
    pulses,
    labels,
    bounds: { top: 2.35, bottom: -2.35, radius: 2.9 },
  }
}
