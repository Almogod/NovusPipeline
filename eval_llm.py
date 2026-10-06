"""
eval_llm.py — NovusPipeline: A/B benchmark of the fine-tuned LoRA adapter

Runs each legacy snippet through the real client stack (local_llm -> llm_worker)
twice — adapter enabled and adapter disabled (same base weights, greedy
decoding) — plus the rule-based modernizer, and scores every output on:

  parses         a Python code block could be extracted and parsed
  valid          passes local_llm.validate_proposal (no new smells, API kept)
  resolved       share of the snippet's rule-engine smells that are gone
  api_kept       every public top-level name of the original still exists
  parity         original and rewrite return the same results (or raise the same
                 exception type) on the snippet's probe calls. Generated code is
                 executed in an isolated subprocess (python -I, temp dir, timeout).
  tokens/s, seconds

Usage:
    python eval_llm.py            # writes .llm_eval/report.json and prints a summary
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import local_llm
from modernizer import CodeModernizer, LegacySmellDetector, top_level_names

GUIDELINES = (
    "- Replace urllib2 with urllib.request (or httpx).\n"
    "- Replace bare `except:` with specific exception types.\n"
    "- Replace os.popen with subprocess.run.\n"
    "- Avoid pickle for untrusted data; prefer json.\n"
    "- Add type hints to every function signature (Python 3.10+ syntax).\n"
    "- Convert Python 2 print statements to print()."
)

CASES: List[Dict[str, Any]] = [
    {"name": "bare_except_parse", "probes": ["to_int('5')", "to_int('x')", "to_int(None)", "to_int('7', 3)"],
     "code": "def to_int(s, default=0):\n    try:\n        return int(s)\n    except:\n        return default\n"},
    {"name": "untyped_math", "probes": ["mean([1, 2, 3])", "mean([])", "clamp(5, 0, 3)", "clamp(-1, 0, 3)"],
     "code": "def mean(values):\n    if not values:\n        return 0.0\n    return sum(values) / len(values)\n\n\n"
             "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n"},
    {"name": "word_freq", "probes": ["word_freq('a b A c b')", "word_freq('')"],
     "code": "def word_freq(text):\n    counts = {}\n    for w in text.lower().split():\n"
             "        counts[w] = counts.get(w, 0) + 1\n    return counts\n"},
    {"name": "untyped_class", "probes": ["(lambda c: (c.add('x'), c.add('y'), c.add('x'), c.top(1)))(Counter())"],
     "code": "class Counter:\n    def __init__(self):\n        self.counts = {}\n\n    def add(self, key):\n"
             "        self.counts[key] = self.counts.get(key, 0) + 1\n        return self.counts[key]\n\n"
             "    def top(self, n=1):\n        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]\n"},
    {"name": "pickle_cache", "probes": ["(save({'a': [1, 2]}, 'cache.bin'), load('cache.bin'))[1]"],
     "code": "import pickle\n\n\ndef save(obj, path):\n    with open(path, 'wb') as f:\n        pickle.dump(obj, f)\n\n\n"
             "def load(path):\n    with open(path, 'rb') as f:\n        return pickle.load(f)\n"},
    {"name": "urllib2_fetch", "probes": [],
     "code": "import urllib2\n\n\ndef fetch(url, timeout=10):\n    resp = urllib2.urlopen(url, timeout=timeout)\n"
             "    return resp.read()\n"},
    {"name": "os_popen", "probes": [],
     "code": "import os\n\n\ndef git_branch():\n    return os.popen('git rev-parse --abbrev-ref HEAD').read().strip()\n"},
    {"name": "py2_mixed", "probes": [],
     "code": "def report(items):\n    total = 0\n    for name, qty in items:\n        print \"%s: %d\" % (name, qty)\n"
             "        total += qty\n    try:\n        return total / len(items)\n    except:\n        return 0\n"},
]

_PARITY_RUNNER = r"""
import json, sys
src_a, src_b, probes = json.load(sys.stdin)
def run(src):
    ns = {}
    try:
        exec(compile(src, "candidate.py", "exec"), ns)
    except BaseException as e:
        return ["load-error", type(e).__name__] * len(probes)
    out = []
    for p in probes:
        try:
            out.append(["ok", repr(eval(p, ns))])
        except BaseException as e:
            out.append(["raise", type(e).__name__])
    return out
print(json.dumps([run(src_a), run(src_b)]))
"""


def parity(original: str, candidate: Optional[str], probes: List[str]) -> Optional[float]:
    """Share of probe calls with identical results; None if the case has no probes."""
    if not probes:
        return None
    if candidate is None:
        return 0.0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            res = subprocess.run([sys.executable, "-I", "-c", _PARITY_RUNNER], input=json.dumps([original, candidate, probes]),
                                 capture_output=True, text=True, cwd=tmp, timeout=30)
            a, b = json.loads(res.stdout)
        except (subprocess.TimeoutExpired, ValueError):
            return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(probes)


def score(case: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
    code = proposal["code"]
    before = Counter(f["smell_id"] for f in LegacySmellDetector.scan_code(case["code"], "x.py"))
    after = Counter(f["smell_id"] for f in LegacySmellDetector.scan_code(code, "x.py")) if code else before
    public = {n for n in (top_level_names(case["code"]) or set()) if not n.startswith("_")}
    total = sum(before.values())
    return {
        "parses": code is not None,
        "valid": proposal["validation"]["valid"],
        "resolved": round(sum(max(0, before[s] - after.get(s, 0)) for s in before) / total, 3) if total else None,
        "api_kept": bool(code) and public <= (top_level_names(code) or set()),
        "parity": parity(case["code"], code, case["probes"]),
        "seconds": proposal["stats"].get("seconds"),
        "tokens_per_second": proposal["stats"].get("tokens_per_second"),
        "new_tokens": proposal["stats"].get("new_tokens"),
    }


def main() -> None:
    os.environ.pop("NOVUS_FAST_TEST", None)
    info = local_llm.get_model_info(probe=True)
    if info.get("load_error"):
        sys.exit(f"LLM worker could not load the model: {info['load_error']}")
    print(f"[eval_llm] worker: {info.get('worker')}", file=sys.stderr)

    results: Dict[str, Any] = {"model": info, "cases": []}
    for case in CASES:
        required = sorted(n for n in (top_level_names(case["code"]) or set()) if not n.startswith("_"))
        row: Dict[str, Any] = {"name": case["name"]}
        outputs = {}
        for engine, adapter in (("adapter", True), ("base", False)):
            t = time.perf_counter()
            p = local_llm.propose_modernization(case["code"], GUIDELINES, required_names=required, adapter=adapter)
            if p["engine"] == "rule-based":
                sys.exit(f"worker fell back to rules for {case['name']}: {p['error']}")
            row[engine] = score(case, p)
            row[engine]["wall_seconds"] = round(time.perf_counter() - t, 1)
            row[engine]["issues"] = p["validation"]["issues"]
            outputs[engine] = p["raw"]
            print(f"[eval_llm] {case['name']:18s} {engine:7s} {row[engine]}", file=sys.stderr)
        rule_code, _ = CodeModernizer.modernize_python(case["code"])
        row["rule_based"] = score(case, {"code": rule_code, "stats": {},
                                         "validation": local_llm.validate_proposal(case["code"], rule_code, required)})
        row["adapter_output_differs_from_base"] = outputs["adapter"] != outputs["base"]
        row["outputs"] = outputs
        results["cases"].append(row)

    summary = {}
    for engine in ("adapter", "base", "rule_based"):
        rows = [c[engine] for c in results["cases"]]
        mean = lambda key: round(sum(r[key] for r in rows if r[key] is not None) /
                                 max(1, sum(r[key] is not None for r in rows)), 3)
        summary[engine] = {"parses": mean("parses"), "valid": mean("valid"), "resolved": mean("resolved"),
                           "api_kept": mean("api_kept"), "parity": mean("parity")}
        if engine != "rule_based":
            summary[engine]["tokens_per_second"] = mean("tokens_per_second")
    summary["adapter_changes_output_in"] = f"{sum(c['adapter_output_differs_from_base'] for c in results['cases'])}/{len(CASES)} cases"
    results["summary"] = summary

    os.makedirs(".llm_eval", exist_ok=True)
    with open(os.path.join(".llm_eval", "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
