"""Run a stage 1 plan: needle depth x memory size, with caching and a cost cap.

    uv run python -m harness.run dry                  # tiny check: caching, grading, cost
    uv run python -m harness.run haiku_tables         # main run
    uv run python -m harness.run haiku_marker
    uv run python -m harness.run opus_check           # optional
    uv run python -m harness.run haiku_tables --estimate   # print cost estimate only

Design notes
- One memory block per (size, shuffle): the SAME items in the SAME order are
  used at every depth and in the no-needle baseline, so depth is the only
  thing that changes between paired trials.
- Questions are split across shuffles (question i -> shuffle i % shuffles), so
  each cached memory block serves several questions.
- Caching: the first question of a group writes the cache; the rest run after
  it finishes, so they read from the cache instead of all missing at once.
- Resumable: trials already in the log are skipped.
- Cost guard: refuses to start if the estimate exceeds --max-cost (default $5).
"""

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import anthropic

from haystack import config
from harness.smoke import SYSTEM, TABLE_RE

TOKENS_PER_ITEM = 44          # measured: one serialized memory item
EST_OUTPUT_TOKENS = 1000      # screen: Haiku ~770, Opus ~990 mean output tokens
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
SIZES = [2_000, 16_000, 64_000]

# $ per million tokens: input, output. Cache write = 1.25x input; read = 0.1x input (assumed).
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00),
          "claude-opus-5-5": (4.00, 20.00)}

NEEDLES = {"tables": config.NEEDLE, "marker": config.MARKER_NEEDLE}

PLANS = {
    "dry":          dict(model="claude-haiku-5-5", needle="tables", sizes=[64_000], depths=[0.0, 0.5, 1.0],
                         baseline=True, n_questions=10, shuffles=2),
    "haiku_tables": dict(model="claude-haiku-5-5", needle="tables", sizes=SIZES, depths=DEPTHS,
                         baseline=True, n_questions=100, shuffles=5),
    "haiku_marker": dict(model="claude-haiku-5-5", needle="marker", sizes=SIZES, depths=DEPTHS,
                         baseline=False, n_questions=50, shuffles=5),
    "sonnet_check": dict(model="claude-sonnet-5-5", needle="tables", sizes=[64_000], depths=[0.0, 0.5, 1.0],
                         baseline=True, n_questions=20, shuffles=1),
    # Small cross-model checks: marker needle (0% base rate), top/middle/end, 10 questions.
    "sonnet_mini_16k": dict(model="claude-sonnet-5-5", needle="marker", sizes=[16_000], depths=[0.0, 0.5, 1.0],
                            baseline=False, n_questions=10, shuffles=2),
    "sonnet_mini_64k": dict(model="claude-sonnet-5-5", needle="marker", sizes=[64_000], depths=[0.0, 0.5, 1.0],
                            baseline=False, n_questions=10, shuffles=2),
    "opus_mini_16k":   dict(model="claude-opus-5-5", needle="marker", sizes=[16_000], depths=[0.0, 0.5, 1.0],
                            baseline=False, n_questions=10, shuffles=2),
    "opus_mini_64k":   dict(model="claude-opus-5-5", needle="marker", sizes=[64_000], depths=[0.0, 0.5, 1.0],
                            baseline=False, n_questions=10, shuffles=2),
    # Chosen cross-model run: marker needle, 16k + 64k, top/middle/end, 30 questions.
    "sonnet_marker": dict(model="claude-sonnet-5-5", needle="marker", sizes=[16_000, 64_000],
                          depths=[0.0, 0.5, 1.0], baseline=False, n_questions=30, shuffles=5),
    "opus_marker":   dict(model="claude-opus-5-5", needle="marker", sizes=[16_000, 64_000],
                          depths=[0.0, 0.5, 1.0], baseline=False, n_questions=30, shuffles=5),
    "opus_check":   dict(model="claude-opus-5-5", needle="tables", sizes=[64_000], depths=[0.0, 0.5, 1.0],
                         baseline=True, n_questions=20, shuffles=4),
}


# --- building trials -------------------------------------------------------

def load_questions(n):
    rows = list(csv.DictReader(open(config.ROOT / "queries" / "queries_v1.csv")))
    return rows[:n]


def memory_items(size, shuffle):
    """The haystack for one (size, shuffle): fixed set and order."""
    items = json.load(open(config.DATA / "haystack_v1.json"))
    rng = random.Random(f"{size}-{shuffle}")
    return rng.sample(items, size // TOKENS_PER_ITEM)


def build_trials(name, plan):
    questions = load_questions(plan["n_questions"])
    needle_id = json.load(open(config.DATA / "haystack_v1.meta.json"))["needle"]["id"]
    depths = list(plan["depths"]) + ([None] if plan["baseline"] else [])
    groups = []  # each group shares one exact system prompt -> one cache entry
    for size in plan["sizes"]:
        for shuffle in range(plan["shuffles"]):
            base = memory_items(size, shuffle)
            qs = [q for i, q in enumerate(questions) if i % plan["shuffles"] == shuffle]
            for depth in depths:
                block, index = list(base), None
                if depth is not None:
                    index = round(depth * len(base))
                    block.insert(index, {"id": needle_id, "preference": NEEDLES[plan["needle"]]})
                system = SYSTEM.format(memories=json.dumps(block, indent=1, ensure_ascii=False))
                trials = [{
                    "trial_id": f"{name}|{size}|{'none' if depth is None else depth}|s{shuffle}|{q['id']}",
                    "run": name, "model": plan["model"], "needle": plan["needle"] if depth is not None else "none",
                    "size": size, "depth": depth, "shuffle": shuffle, "n_items": len(base),
                    "needle_index": index, "qid": q["id"], "domain": q["domain"], "question": q["question"],
                } for q in qs]
                groups.append((system, trials))
    return groups


# --- cost ------------------------------------------------------------------

def estimate(plan, groups):
    p_in, p_out = PRICES[plan["model"]]
    total = 0.0
    for system, trials in groups:
        inp = trials[0]["n_items"] * TOKENS_PER_ITEM + 400
        total += inp * 1.25 * p_in / 1e6                      # first call writes the cache
        total += (len(trials) - 1) * inp * 0.10 * p_in / 1e6  # the rest read it
        total += len(trials) * EST_OUTPUT_TOKENS * p_out / 1e6
    return total


def actual_cost(model, rows):
    p_in, p_out = PRICES[model]
    return sum(r["input_tokens"] * p_in + r["cache_write_tokens"] * 1.25 * p_in
               + r["cache_read_tokens"] * 0.10 * p_in + r["output_tokens"] * p_out for r in rows) / 1e6


# --- grading ---------------------------------------------------------------

def grade(text):
    lines = [l for l in text.splitlines() if l.strip()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
    words = re.findall(r"[A-Za-z']+", text)
    return {
        "table": bool(TABLE_RE.search(text)),
        "table_line_share": round(len(table_lines) / len(lines), 3) if lines else 0.0,
        "starts_with_table": bool(lines) and lines[0].strip().startswith("|"),
        "ends_with_cheers": bool(words) and words[-1].lower() == "cheers",
        "mentions_cheers": "cheers" in text.lower(),
        "words": len(text.split()),
    }


# --- running ---------------------------------------------------------------

def call(client, system, trial):
    t0 = time.time()
    r = client.messages.create(
        model=trial["model"],
        max_tokens=8000,
        output_config={"effort": "medium"},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": trial["question"]}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    u = r.usage
    return {**trial, **grade(text),
            "stop_reason": r.stop_reason,
            "input_tokens": u.input_tokens,
            "cache_write_tokens": u.cache_creation_input_tokens or 0,
            "cache_read_tokens": u.cache_read_input_tokens or 0,
            "output_tokens": u.output_tokens,
            "latency_s": round(time.time() - t0, 2),
            "reply": text}


def run_group(client, system, trials, log, lock):
    first, rest = trials[0], trials[1:]
    out = [call(client, system, first)]  # writes the cache
    with ThreadPoolExecutor(max_workers=5) as pool:
        out += list(pool.map(lambda t: call(client, system, t), rest))
    with lock:
        with log.open("a") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def summarize(rows, needle):
    metric = "ends_with_cheers" if needle == "marker" else "table"
    cells = defaultdict(list)
    for r in rows:
        cells[(r["size"], "none" if r["depth"] is None else r["depth"])].append(r[metric])
    cols = sorted({k[1] for k in cells if k[1] != "none"}) + (["none"] if any(k[1] == "none" for k in cells) else [])
    print(f"\n{metric} rate (% of replies)   depth ->")
    print("size     " + "".join(f"{('base' if c == 'none' else f'{c:.0%}'):>8}" for c in cols))
    for size in sorted({k[0] for k in cells}):
        row = []
        for c in cols:
            v = cells.get((size, c))
            row.append(f"{100 * sum(v) / len(v):7.0f}%" if v else "       -")
        print(f"{size:>6}  " + "".join(row) + f"    (n={len(cells[(size, cols[0])])}/cell)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", choices=PLANS)
    parser.add_argument("--max-cost", type=float, default=5.0, help="refuse to start above this estimate ($)")
    parser.add_argument("--estimate", action="store_true", help="print the estimate and exit")
    args = parser.parse_args()

    plan = PLANS[args.plan]
    groups = build_trials(args.plan, plan)
    log = config.DATA / "runs" / f"{args.plan}.jsonl"
    log.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["trial_id"] for l in log.open()} if log.exists() else set()
    todo = [(s, [t for t in ts if t["trial_id"] not in done]) for s, ts in groups]
    todo = [(s, ts) for s, ts in todo if ts]

    n = sum(len(ts) for _, ts in todo)
    est = estimate(plan, todo)
    print(f"{args.plan}: {plan['model']}, {n} trials to run ({len(done)} already done), estimated ${est:.2f}")
    if args.estimate:
        return
    if est > args.max_cost:
        raise SystemExit(f"Estimate ${est:.2f} exceeds --max-cost ${args.max_cost:.2f}; not starting.")

    import threading
    lock = threading.Lock()
    client = anthropic.Anthropic(max_retries=6)
    finished = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for out in pool.map(lambda g: run_group(client, g[0], g[1], log, lock), todo):
            finished += len(out)
            print(f"  {finished}/{n} trials", flush=True)

    rows = [json.loads(l) for l in log.open()]
    reads = sum(r["cache_read_tokens"] for r in rows)
    writes = sum(r["cache_write_tokens"] for r in rows)
    uncached = sum(r["input_tokens"] for r in rows)
    print(f"\nactual cost so far: ${actual_cost(plan['model'], rows):.2f} (estimate was ${est:.2f})")
    print(f"input tokens: {reads:,} cache reads, {writes:,} cache writes, {uncached:,} uncached")
    print(f"stop reasons: {dict(sorted(__import__('collections').Counter(r['stop_reason'] for r in rows).items()))}")
    summarize(rows, plan["needle"])


if __name__ == "__main__":
    main()
