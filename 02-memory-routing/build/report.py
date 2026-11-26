"""Summarize a run: one row per setup, and adherence by setup x scenario.

    uv run python -m build.report pilot
"""

import argparse
import json
from collections import defaultdict

from . import config

ORDER = ["none", "D", "A", "E", "B1", "G", "C-types", "C-embed", "C-Jev", "C1", "C2", "C3"]


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (config.DATA / "runs" / f"{args.plan}.jsonl").read_text().splitlines()]
    gold = [r for r in rows if r["scenario"] != "no_memory"]
    nomem = [r for r in rows if r["scenario"] == "no_memory"]

    by = defaultdict(list)
    for r in gold:
        by[r["setup"]].append(r)
    nm = defaultdict(list)
    for r in nomem:
        nm[r["setup"]].append(r)

    print(f"{args.plan}: {len(rows)} trials\n")
    print("| Setup | Adherence | Planted gold in prompt | Any gold in prompt | Passed with no gold | "
          "Read failures | Gold group searched | No-memory questions clean | Input tokens / trial |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in ORDER:
        rs = by.get(s, [])
        if not rs:
            continue
        n = len(rs)
        passed = sum(r["passes"] for r in rs)
        planted = sum(r["gold_in_context"] for r in rs)
        anyg = sum(r.get("any_gold_in_context", r["gold_in_context"]) for r in rs)
        no_gold = [r for r in rs if not r.get("any_gold_in_context", r["gold_in_context"])]
        pass_no_gold = sum(r["passes"] for r in no_gold)
        read_fail = sum(1 for r in rs if r["gold_in_context"] and not r["passes"])
        grp = [r for r in rs if "gold_group_searched" in r]
        nmr = nm.get(s, [])
        tok = sum(r["input_tokens"] + (r.get("cache_read_input_tokens") or 0) + (r.get("cache_creation_input_tokens") or 0)
                  for r in rs) / n
        print(f"| {s} | {pct(passed, n)} | {pct(planted, n)} | {pct(anyg, n)} | "
              f"{pass_no_gold}/{len(no_gold)} | {read_fail}/{planted} | "
              f"{pct(sum(r['gold_group_searched'] for r in grp), len(grp)) if grp else '-'} | "
              f"{pct(sum(r['passes'] for r in nmr), len(nmr)) if nmr else '-'} | {tok:,.0f} |")

    scen = sorted({r["scenario"] for r in gold})
    print("\nAdherence by scenario\n")
    print("| Setup | " + " | ".join(scen) + " |")
    print("|---|" + "---|" * len(scen))
    for s in ORDER:
        rs = by.get(s, [])
        if not rs:
            continue
        cells = []
        for sc in scen:
            x = [r for r in rs if r["scenario"] == sc]
            cells.append(f"{sum(r['passes'] for r in x)}/{len(x)}" if x else "-")
        print(f"| {s} | " + " | ".join(cells) + " |")

    crit = defaultdict(lambda: defaultdict(int))
    for r in gold:
        if r.get("critic"):
            crit[r["setup"]][r["critic"]["label"]] += 1
    print("\nReasoning critic, when the planted gold was in the prompt\n")
    print("| Setup | not_mentioned | mentioned_set_aside | mentioned_applied |")
    print("|---|---|---|---|")
    for s in ORDER:
        c = crit.get(s)
        if c:
            print(f"| {s} | {c['not_mentioned']} | {c['mentioned_set_aside']} | {c['mentioned_applied']} |")

    prices = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
    batched = args.plan.startswith(("main_", "swap_", "smoke_engine"))

    def cost(r):
        p_in, p_out = prices[r.get("model", "claude-haiku-5-5")]
        c = (r["input_tokens"] * p_in + (r.get("cache_read_input_tokens") or 0) * p_in * 0.1
             + (r.get("cache_creation_input_tokens") or 0) * p_in * 1.25 + r["output_tokens"] * p_out) / 1e6
        c *= 0.5 if batched else 1.0
        if r.get("judge_tokens"):
            a, b = r["judge_tokens"]
            c += (a * 2 + b * 10) / 1e6 if r.get("grade_method") == "word_list+sonnet_judge" else (a * 0.10 + b * 0.50) / 1e6
        if r.get("sonnet_judge_tokens"):
            a, b = r["sonnet_judge_tokens"]
            c += (a * 2 + b * 10) / 1e6
        if r.get("critic"):
            a, b = r["critic"]["tokens"]
            c += (a * 0.10 + b * 0.50) / 1e6
        return c
    print(f"\nClaude cost of this run ({'batched' if batched else 'standard'} rate, incl. judging and critic): ${sum(map(cost, rows)):.2f}")


if __name__ == "__main__":
    main()
