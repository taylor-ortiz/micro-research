"""Adherence with 95% confidence intervals, and paired differences between setups.

    uv run python -m build.stats haiku     # main_haiku + jevplus_haiku
    uv run python -m build.stats sonnet    # main_sonnet + jevplus_sonnet

Only gold questions are counted (no-memory questions are reported separately as
over-application). Intervals come from a bootstrap that resamples whole questions
(2,000 draws), so the repeated depths of one question are not treated as independent.
Paired differences compare two setups on exactly the same (question, depth) trials.
"""

import argparse
import json
import random
from collections import defaultdict

from . import config

RUNS = {"haiku": ["main_haiku", "jevplus_haiku"], "sonnet": ["main_sonnet", "jevplus_sonnet"]}
ORDER = ["D", "A", "C-Jev+", "G", "C-Jev", "B1", "C-types", "C1", "C2", "C3", "C-embed", "E", "none"]
DRAWS = 2000


def tokens(r):
    return (r["input_tokens"] + (r.get("cache_read_input_tokens") or 0) + (r.get("cache_creation_input_tokens") or 0))


def load(model):
    rows = []
    for plan in RUNS[model]:
        path = config.DATA / "runs" / f"{plan}.jsonl"
        if path.exists():
            rows += [json.loads(l) for l in path.read_text().splitlines()]
    return rows


def boot(by_item, rng):
    """by_item: {item: [0/1, ...]} -> (mean, low, high), resampling items."""
    items = list(by_item)
    flat = [v for vs in by_item.values() for v in vs]
    est = sum(flat) / len(flat)
    means = []
    for _ in range(DRAWS):
        pick = [rng.choice(items) for _ in items]
        vals = [v for i in pick for v in by_item[i]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return est, means[int(0.025 * DRAWS)], means[int(0.975 * DRAWS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=RUNS)
    ap.add_argument("--reference", nargs="+", default=["A", "G", "E"])
    args = ap.parse_args()
    rows = [r for r in load(args.model) if r["version"] == "gold"]
    rng = random.Random(0)
    gold = [r for r in rows if r["scenario"] != "no_memory"]
    cell = {(r["setup"], r["item"], r["depth"]): int(r["passes"]) for r in gold}
    a_tok = None

    print(f"{args.model}: {len(gold)} gold trials\n")
    print("| Setup | Adherence | 95% CI | Gold in prompt | Tokens / question | Trials |")
    print("|---|---|---|---|---|---|")
    summary = {}
    for s in ORDER:
        rs = [r for r in gold if r["setup"] == s]
        if not rs:
            continue
        by = defaultdict(list)
        for r in rs:
            by[r["item"]].append(int(r["passes"]))
        est, lo, hi = boot(by, rng)
        tok = sum(map(tokens, rs)) / len(rs)
        if s == "A":
            a_tok = tok
        gin = sum(r["gold_in_context"] for r in rs) / len(rs)
        summary[s] = {"adherence": est, "ci": [lo, hi], "tokens": tok, "gold_in_prompt": gin, "n": len(rs)}
        print(f"| {s} | {est:.0%} | {lo:.0%}–{hi:.0%} | {gin:.0%} | {tok / 1000:.1f}k | {len(rs)} |")
    if a_tok:
        print("\nTokens relative to A: " + ", ".join(f"{s} {100 * (1 - v['tokens'] / a_tok):.0f}% fewer"
                                                for s, v in summary.items() if s not in ("A", "none")))

    print("\nPaired differences on the same trials (setup minus reference, points, 95% CI)\n")
    refs = [x for x in args.reference if x in summary]
    print("| Setup | " + " | ".join(f"vs {x}" for x in refs) + " |")
    print("|---|" + "---|" * len(refs))
    for s in ORDER:
        if s not in summary or s in ("none",):
            continue
        cells = []
        for ref in refs:
            if ref == s:
                cells.append("—")
                continue
            by = defaultdict(list)
            for (ss, item, depth), v in cell.items():
                if ss == s and (ref, item, depth) in cell:
                    by[item].append(v - cell[(ref, item, depth)])
            if not by:
                cells.append("n/a")
                continue
            est, lo, hi = boot(by, rng)
            sig = "*" if lo > 0 or hi < 0 else ""
            cells.append(f"{100 * est:+.0f} ({100 * lo:+.0f} to {100 * hi:+.0f}){sig}")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print("\n* = the 95% interval excludes zero")

    nm = [r for r in rows if r["scenario"] == "no_memory"]
    if nm:
        print("\nNo-memory questions answered cleanly (no irrelevant memory forced in)")
        for s in ORDER:
            x = [r for r in nm if r["setup"] == s]
            if x:
                print(f"  {s:8s} {sum(r['passes'] for r in x)}/{len(x)}")
    out = config.DATA / f"stats_{args.model}.json"
    out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
