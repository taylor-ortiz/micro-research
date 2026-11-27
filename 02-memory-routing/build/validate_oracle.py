"""Validate each question under perfect retrieval with both versions of its memory.

    uv run python -m build.validate_oracle

Each gold item runs with only its gold memory (2 samples) and only its twin
memory (2 samples). It is kept only if all four replies follow the memory they
were given. Raw rows: data/validate/raw.jsonl (resumable). Verdicts:
data/validate/results.json.
"""

import json
import random
import threading

from . import config, harness
from .grade import grade

DIR = config.DATA / "validate"
RAW = DIR / "raw.jsonl"
_lock = threading.Lock()


def key(j):
    return f"{j['id']}:{j['version']}:{j['s']}"


def run_one(item, j, out):
    tools = None
    if item.get("tools"):
        tools = list(item["tools"])
        random.Random(key(j)).shuffle(tools)
    mem = [{"id": "m0001", "memory": item[f"{j['version']}_memory"]}]
    r = harness.answer(item["query"], memories=mem, tools=tools)
    g = grade(item, r, j["version"])
    row = {**j, **r, "passes": g["passes"], "method": g["method"], "judge_tokens": g["judge_tokens"]}
    with _lock:
        out.write(json.dumps(row) + "\n")
        out.flush()
    return row


def main():
    items = [json.loads(l) for l in (config.DATA / "items_v1.jsonl").read_text().splitlines()]
    items = {it["id"]: it for it in items if it["gold_memory"]}
    DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            row = json.loads(l)
            done[key(row)] = row
    jobs = [dict(id=i, version=v, s=s) for i in items for v in ("gold", "twin") for s in range(2)]
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(todo)} oracle calls to run ({len(done)} cached)")
    with RAW.open("a") as out:
        for row in harness.pmap(lambda j: run_one(items[j["id"]], j, out), todo):
            done[key(row)] = row
    rows = list(done.values())

    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    results, summary = {}, {}
    for i, rs in by.items():
        gold = [r["passes"] for r in rs if r["version"] == "gold"]
        twin = [r["passes"] for r in rs if r["version"] == "twin"]
        keep = all(gold) and all(twin)
        results[i] = {"gold": gold, "twin": twin, "keep": keep}
        sc = items[i]["scenario"]
        summary.setdefault(sc, [0, 0])
        summary[sc][0] += keep
        summary[sc][1] += 1
    judge_cost = sum((t[0] * 2 + t[1] * 10) if "sonnet" in r["method"] else (t[0] * 0.10 + t[1] * 0.50)
                     for r in rows if (t := r["judge_tokens"])) / 1e6
    (DIR / "results.json").write_text(json.dumps({"summary": summary, "items": results}, indent=1))
    for sc, (k, n) in sorted(summary.items()):
        print(f"{sc:16s} kept {k:3d}/{n}")
    print(f"cost: answers ${harness.cost(rows):.2f}, judge ${judge_cost:.2f}")


if __name__ == "__main__":
    main()
