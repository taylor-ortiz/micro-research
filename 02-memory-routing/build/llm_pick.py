"""LLM memory picker: Haiku chooses the memories, as Jev does in C-Jev+.

Two setups use it:
  L-cat  the same candidates as C-Jev+ (every memory in Jev's chosen groups, cutoff 0.1)
  L-all  every memory in the store, no group step
Haiku reads the candidates 100 at a time next to the user's message and scores each
memory that could change or improve the reply from 0 to 1 (unlisted = 0). Memories at
the tuned cutoff or above go in, at most 20, highest first, in shuffle order, the same
rule as C-Jev+. Picks depend only on the question and version, so they are cached in
data/llm_picks/ and reused across depths and answering models.

    uv run python -m build.llm_pick      # tune the cutoff on the dev split
"""

import json
import threading
import time
from collections import defaultdict

from pydantic import BaseModel

from . import config, harness
from .run import ITEMS, corpus, jev_scores

DIR = config.DATA / "llm_picks"
CHUNK = 100
GROUP_CUTOFF = 0.1
PICKER_MODEL = config.JUDGE_MODEL   # Haiku 5.5
_lock = threading.Lock()

SYSTEM = """You select memories for an AI assistant. You get a user's message and a numbered \
list of stored memories about the user. For each memory that could change or improve the \
assistant's reply to this message, give its number and a score from 0 to 1 for how likely it \
is to matter. Leave out memories that would not matter; most should be left out."""


class Pick(BaseModel):
    number: int
    score: float


class Picks(BaseModel):
    picks: list[Pick]


def pick(query, mems):
    """Score memories with Haiku. Returns ({memory id: score}, input tokens, output tokens, seconds)."""
    t0 = time.time()
    chunks = [mems[i:i + CHUNK] for i in range(0, len(mems), CHUNK)]

    def one(chunk):
        listing = "\n".join(f"{n + 1}. {m['memory']}" for n, m in enumerate(chunk))
        r = harness.client().messages.parse(
            model=PICKER_MODEL, max_tokens=4000, system=SYSTEM, output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"User message: {query}\n\nMemories:\n{listing}"}],
            output_format=Picks)
        scores = {chunk[p.number - 1]["id"]: max(0.0, min(1.0, p.score))
                  for p in r.parsed_output.picks if 1 <= p.number <= len(chunk)}
        return scores, r.usage.input_tokens, r.usage.output_tokens

    out, tin, tout = {}, 0, 0
    for scores, a, b in harness.pmap(one, chunks, workers=len(chunks) or 1):
        out.update(scores)
        tin += a
        tout += b
    return out, tin, tout, time.time() - t0


def candidates(setup, item, shuffle, version):
    rows = corpus(item, shuffle, version)
    if setup == "L-all":
        return rows, None
    scores, _, _ = jev_scores(item["query"])
    chosen = [g for g, p in scores.items() if p >= GROUP_CUTOFF]
    return [r for r in rows if r["group"] in chosen], chosen


def cached_pick(setup, item, shuffle, version):
    """Picks for one (setup, question, version), keyed by stable memory keys."""
    path = DIR / f"{setup}.jsonl"
    key = f"{item['id']}|{version}"
    with _lock:
        if path.exists():
            for l in path.read_text().splitlines():
                row = json.loads(l)
                if row["key"] == key:
                    return row
    rows, chosen = candidates(setup, item, shuffle, version)
    scores, tin, tout, secs = pick(item["query"], rows)
    by_key = {r["key"]: scores.get(r["id"], 0.0) for r in rows}
    row = {"key": key, "scores": by_key, "n_candidates": len(rows), "groups": chosen,
           "picker_input_tokens": tin, "picker_output_tokens": tout, "picker_seconds": round(secs, 2)}
    DIR.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    from .trials import shuffle_of
    dev = json.loads((config.DATA / "jev_tune" / "dev_split.json").read_text())
    summary = {}
    for setup in ("L-cat", "L-all"):
        rows = harness.pmap(lambda i: (i, cached_pick(setup, ITEMS[i], shuffle_of(i), "gold")), dev, workers=6)
        n = len(rows)
        cost = sum(r["picker_input_tokens"] * 0.10 + r["picker_output_tokens"] * 0.50 for _, r in rows) / 1e6
        secs = sum(r["picker_seconds"] for _, r in rows) / n
        print(f"\n{setup}: {n} dev questions, ~{sum(r['n_candidates'] for _, r in rows) / n:.0f} candidates each, "
              f"picker ${cost:.2f}, {secs:.1f} s per question")
        curve = []
        for c in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
            hit = sum(r["scores"].get("GOLD", 0) >= c for _, r in rows) / n
            kept = sum(sum(v >= c for v in r["scores"].values()) for _, r in rows) / n
            curve.append({"cutoff": c, "gold_kept": hit, "avg_kept": kept})
            print(f"  score >= {c}: gold kept {hit:.0%}, memories kept {kept:.1f}")
        by = defaultdict(list)
        for i, r in rows:
            by[ITEMS[i]["scenario"]].append(r["scores"].get("GOLD", 0) >= 0.7)
        print("  gold kept at 0.7 by scenario: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by.items())))
        summary[setup] = {"curve": curve, "picker_cost_dev": cost, "seconds_per_question": secs}
    (DIR / "dev_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
