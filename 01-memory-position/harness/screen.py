"""Screen draft questions with no memory, then pick the final 100.

A question is dropped if ANY no-memory reply contains a Markdown table:
the needle must be the only thing that produces tables. From the survivors,
100 are chosen keeping the topic mix as even as possible.

    uv run python -m harness.screen
"""

import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import anthropic

from haystack import config
from harness.smoke import SYSTEM, TABLE_RE

DRAFT = config.ROOT / "queries" / "queries_draft.csv"
FINAL = config.ROOT / "queries" / "queries_v1.csv"
RESULTS = config.DATA / "screen_results.json"
SAMPLES = {"claude-haiku-5-5": 2, "claude-opus-5-5": 1}
TARGET = 100
SEED = 20261123


def ask(client, model, question):
    r = client.messages.create(
        model=model,
        max_tokens=8000,
        output_config={"effort": "medium"},
        system=SYSTEM.format(memories="[]"),
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    return {"table": bool(TABLE_RE.search(text)), "words": len(text.split()),
            "stop_reason": r.stop_reason, "output_tokens": r.usage.output_tokens}


def main():
    questions = list(csv.DictReader(open(DRAFT)))
    client = anthropic.Anthropic(max_retries=6)
    jobs = [(q["id"], m, q["question"]) for q in questions for m, n in SAMPLES.items() for _ in range(n)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(lambda j: ask(client, j[1], j[2]), jobs))

    by_q = defaultdict(list)
    for (qid, model, _), out in zip(jobs, outs):
        by_q[qid].append({"model": model, **out})
    RESULTS.write_text(json.dumps(by_q, indent=1))

    survivors = [q for q in questions if not any(s["table"] for s in by_q[q["id"]])]
    rng = random.Random(SEED)
    pools = defaultdict(list)
    for q in survivors:
        pools[q["domain"]].append(q)
    for p in pools.values():
        rng.shuffle(p)
    final = []
    while len(final) < min(TARGET, len(survivors)):  # round-robin keeps topics balanced
        for d in sorted(pools):
            if pools[d] and len(final) < TARGET:
                final.append(pools[d].pop())
    final.sort(key=lambda q: q["id"])
    with open(FINAL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "domain", "question"])
        w.writeheader()
        w.writerows(final)

    dropped = [(q["id"], q["question"], [s["model"] for s in by_q[q["id"]] if s["table"]])
               for q in questions if q not in survivors]
    print(f"screened {len(questions)} questions x {sum(SAMPLES.values())} replies")
    print(f"dropped for producing a table with no memory: {len(dropped)}")
    for qid, text, models in dropped:
        print(f"  {qid} {text}  <- {', '.join(models)}")
    print(f"survivors: {len(survivors)}  final: {len(final)} -> {FINAL.relative_to(config.ROOT)}")
    print("final per topic:", dict(sorted(Counter(q['domain'] for q in final).items())))
    for m in SAMPLES:
        rs = [s for v in by_q.values() for s in v if s["model"] == m]
        print(f"{m}: median reply {sorted(s['words'] for s in rs)[len(rs)//2]} words, "
              f"mean output tokens {sum(s['output_tokens'] for s in rs)//len(rs)}, "
              f"non-end_turn stops {sum(s['stop_reason'] != 'end_turn' for s in rs)}")


if __name__ == "__main__":
    main()
