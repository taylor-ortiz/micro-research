"""Check for C-Jev+: does Jev rank the gold memory near the top of the chosen groups?

    uv run python -m build.jev_memories

On the 75 dev questions (excluded from the main analysis): Jev picks groups with the
tuned group question at cutoff 0.1, then answers one yes/no question per memory in
those groups ("could this memory change or improve the reply?"). The gold memory's
rank under Jev is compared with its rank under the shared hybrid search over the
same candidates. No Claude calls. Writes data/jev_memories/{scores.jsonl, summary.json}.
"""

import json
import threading
from collections import defaultdict

import httpx

from . import config, harness, search
from .run import GROUPS, ITEMS, corpus, jev_scores
from .trials import shuffle_of

DIR = config.DATA / "jev_memories"
URL = "https://api.typesafe.ai/v1/systemone"
GROUP_CUTOFF = 0.1
CHUNK = 250
_lock = threading.Lock()
QUESTION = "Could this memory change or improve the assistant's reply to user_message? Memory: {m}"


def score_memories(query, mems):
    out, tokens = {}, 0
    for i in range(0, len(mems), CHUNK):
        chunk = mems[i:i + CHUNK]
        qs = {f"m{j:03d}": {"type": "noul", "instructions": QUESTION.format(m=m["memory"])} for j, m in enumerate(chunk)}
        r = httpx.post(URL, timeout=300, headers={"Authorization": "Bearer proxy-injected"},
                       json={"model": "jev-latest", "state": {"user_message": query}, "questions": qs})
        r.raise_for_status()
        d = r.json()
        for k, v in d["answers"].items():
            out[chunk[int(k[1:])]["id"]] = v["noul"]
        tokens += d.get("usage", {}).get("input_tokens", 0)
    return out, tokens


def check(item_id):
    item = ITEMS[item_id]
    rows = corpus(item, shuffle_of(item_id), "gold")
    gold = next(r for r in rows if r["gold"])
    gscores, _, _ = jev_scores(item["query"])
    chosen = [g for g, p in gscores.items() if p >= GROUP_CUTOFF]
    cands = [r for r in rows if r["group"] in chosen]
    if gold not in cands:
        return {"item": item_id, "scenario": item["scenario"], "gold_group_chosen": False,
                "n_candidates": len(cands), "groups": len(chosen)}
    mscores, tokens = score_memories(item["query"], cands)
    by_jev = sorted(cands, key=lambda r: -mscores[r["id"]])
    _, ranking = search.hybrid_search(item["query"], cands, k=len(cands))
    return {"item": item_id, "scenario": item["scenario"], "gold_group_chosen": True, "groups": len(chosen),
            "n_candidates": len(cands), "jev_rank": [r["id"] for r in by_jev].index(gold["id"]) + 1,
            "search_rank": [x["id"] for x in ranking].index(gold["id"]) + 1,
            "gold_score": mscores[gold["id"]], "scores": mscores, "jev_input_tokens": tokens}


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    dev = json.loads((config.DATA / "jev_tune" / "dev_split.json").read_text())
    rows = harness.pmap(check, dev, workers=6)
    (DIR / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if r["gold_group_chosen"]]
    n = len(rows)
    print(f"{n} dev questions; gold group chosen in {len(ok)}; ~{sum(r['n_candidates'] for r in rows) / n:.0f} candidate memories each")
    print("\n| Gold memory in the top k | Jev ranking | Similarity search ranking |")
    print("|---|---|---|")
    summary = {}
    for k in (1, 3, 5, 10, 20):
        j = sum(r["jev_rank"] <= k for r in ok) / n
        s = sum(r["search_rank"] <= k for r in ok) / n
        summary[f"top{k}"] = {"jev": j, "search": s}
        print(f"| top {k} | {j:.0%} | {s:.0%} |")
    print("\nBy scenario, top 10 (Jev / search):")
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    for sc, rs in sorted(by.items()):
        j = sum(r.get("jev_rank", 9e9) <= 10 for r in rs)
        s = sum(r.get("search_rank", 9e9) <= 10 for r in rs)
        print(f"  {sc:15s} {j}/{len(rs)}  vs  {s}/{len(rs)}")
    print("\nKeep memories by Jev score (instead of a fixed top k):")
    for t in (0.5, 0.7, 0.8, 0.9):
        hit = sum(r["gold_score"] >= t for r in ok) / n
        kept = sum(sum(v >= t for v in r["scores"].values()) for r in ok) / max(len(ok), 1)
        summary[f"score>={t}"] = {"gold_kept": hit, "avg_kept": kept}
        print(f"  score >= {t}: gold kept {hit:.0%}, memories kept {kept:.1f} on average")
    (DIR / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
