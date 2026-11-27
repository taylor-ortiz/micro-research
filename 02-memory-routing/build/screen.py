"""No-memory screens: keep only questions where a pass means the memory was used.

    uv run python -m build.screen

Rules (no memory in the prompt):
- style rules:      gold check passes in <= 20% of 5 samples
- task procedures:  of the target and twin tools, the target is used first in <= 30%
                    of 5 samples (up to 3 tool rounds)
- allergy:          reply names a food on the allergen's list in >= 60% of 5 samples
- fact, unclear type, past session: gold pass rule met in <= 1 of 3 samples (judge)

Raw replies go to data/screen/raw.jsonl (resumable); the verdicts to
data/screen/results.json.
"""

import json
import threading

from . import config, harness, style_rules

DIR = config.DATA / "screen"
RAW = DIR / "raw.jsonl"
GEN = config.DATA / "gen"
_lock = threading.Lock()


def load(kind):
    return [json.loads(l) for l in (GEN / f"{kind}.jsonl").read_text().splitlines()]


def jobs():
    out = []
    for i, r in enumerate(style_rules.RULES):
        out += [dict(kind="style_rule", idx=i, s=s, q=r["question"], tools=None) for s in range(5)]
    for i, t in enumerate(load("task_procedure")):
        tools = [t["target_tool"], t["twin_tool"], *t["other_tools"]]
        out += [dict(kind="task_procedure", idx=i, s=s, q=t["query"], tools=tools) for s in range(5)]
    for i, a in enumerate(load("allergy")):
        out += [dict(kind="allergy", idx=i, s=s, q=a["query"], tools=None) for s in range(5)]
    for kind in ("fact", "unclear_type", "past_session"):
        for i, r in enumerate(load(kind)):
            out += [dict(kind=kind, idx=i, s=s, q=r["query"], tools=None) for s in range(3)]
    return out


def key(j):
    return f"{j['kind']}:{j['idx']}:{j['s']}"


def run_one(j, out):
    # Tools are shuffled per sample so list position can't decide the choice.
    tools = None
    if j["tools"]:
        import random
        tools = list(j["tools"])
        random.Random(key(j)).shuffle(tools)
    r = harness.answer(j["q"], memories=None, tools=tools)
    row = {**j, **r}
    if j["kind"] in ("fact", "unclear_type", "past_session"):
        item = load(j["kind"])[j["idx"]]
        g = harness.judge(j["q"], r["reply"], item["gold_pass"])
        row["gold_pass"] = g["passes"]
        row["judge_tokens"] = [g["input_tokens"], g["output_tokens"]]
    with _lock:
        out.write(json.dumps(row) + "\n")
        out.flush()
    return row


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            row = json.loads(l)
            done[key(row)] = row
    todo = [j for j in jobs() if key(j) not in done]
    print(f"{len(todo)} screen calls to run ({len(done)} cached)")
    with RAW.open("a") as out:
        for row in harness.pmap(lambda j: run_one(j, out), todo):
            done[key(row)] = row
    rows = list(done.values())

    results, kept = {}, {}
    by = {}
    for r in rows:
        by.setdefault((r["kind"], r["idx"]), []).append(r)
    tasks, allergy = load("task_procedure"), load("allergy")
    for (kind, idx), rs in sorted(by.items()):
        n = len(rs)
        if kind == "style_rule":
            rule = style_rules.RULES[idx]
            rate = sum(style_rules.grade(rule, r["reply"], "gold") for r in rs) / n
            twin = sum(style_rules.grade(rule, r["reply"], "twin") for r in rs) / n
            keep = rate <= 0.20
        elif kind == "task_procedure":
            t = tasks[idx]
            def first_of_pair(r):
                pair = [x for x in r.get("tool_trace") or [] if x in (t["target_tool"], t["twin_tool"])]
                return pair[0] if pair else None
            rate = sum(first_of_pair(r) == t["target_tool"] for r in rs) / n
            twin = sum(first_of_pair(r) == t["twin_tool"] for r in rs) / n
            keep = rate <= config.TOOL_SCREEN_MAX_RATE
        elif kind == "allergy":
            a = allergy[idx]
            rate = sum(bool(harness.allergen_hits(r["reply"], a["allergen"])) for r in rs) / n
            twin = sum(bool(harness.allergen_hits(r["reply"], a["twin_allergen"])) for r in rs) / n
            keep = rate >= config.ALLERGY_SCREEN_MIN_RATE
        else:
            rate = sum(r["gold_pass"] for r in rs) / n
            twin = None
            keep = sum(r["gold_pass"] for r in rs) <= 1
        results[f"{kind}:{idx}"] = {"kind": kind, "idx": idx, "gold_rate": rate, "twin_rate": twin, "keep": keep}
        kept.setdefault(kind, [0, 0])
        kept[kind][0] += keep
        kept[kind][1] += 1

    answer_cost = harness.cost(rows)
    judge_cost = sum(r["judge_tokens"][0] * 0.10 + r["judge_tokens"][1] * 0.50 for r in rows if "judge_tokens" in r) / 1e6
    (DIR / "results.json").write_text(json.dumps({"kept": kept, "items": results}, indent=1))
    for kind, (k, n) in kept.items():
        print(f"{kind:16s} kept {k:3d}/{n}")
    print(f"cost: answers ${answer_cost:.2f}, judge ${judge_cost:.2f}")


if __name__ == "__main__":
    main()
