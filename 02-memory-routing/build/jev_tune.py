"""Tune C-Jev on a dev split: which question wording, entry and cutoff find the gold group?

    uv run python -m build.jev_tune

No Haiku answers are needed: each gold memory's group is known, so a variant is
scored by how often Jev picks that group (gold-group recall) and how many groups it
picks on average (cost). The dev split is a stratified third of the gold questions,
including the pilot's; the main analysis is reported on the other two thirds.
Writes data/jev_tune/{dev_split.json, scores.jsonl, summary.json}.
"""

import json
import random
import threading

import httpx

from . import config, harness
from .run import GROUPS, ITEMS, ONTO, entry_text, pilot_items

DIR = config.DATA / "jev_tune"
URL = "https://api.typesafe.ai/v1/systemone"
_lock = threading.Lock()


def c2_text(g):
    return f"{g['name']}. {g['description']}"


QUESTION = {
    "q1": "Should the assistant search this memory group before replying to user_message? Group entry: {e}",
    "q2": ("Could any memory in this group change or improve the assistant's reply to user_message? "
           "Groups that apply to every reply always count. Group entry: {e}"),
}
VARIANTS = {
    "v1_current_c3": ("q1", entry_text, "jev-latest"),
    "v2_reworded_c3": ("q2", entry_text, "jev-latest"),
    "v3_reworded_c2": ("q2", c2_text, "jev-latest"),
    "v4_reworded_c3_preview": ("q2", entry_text, "jev-preview"),
}
CUTOFFS = [round(0.1 * i, 1) for i in range(1, 10)]


def dev_split():
    gold = [it for it in ITEMS.values() if it["gold_memory"]]
    pilot = set(pilot_items())
    by = {}
    for it in gold:
        by.setdefault(it["scenario"], []).append(it["id"])
    rng = random.Random(5)
    dev = set()
    for sc, ids in sorted(by.items()):
        ids = sorted(ids)
        want = round(len(ids) / 3)
        chosen = [i for i in ids if i in pilot]
        rest = [i for i in ids if i not in pilot]
        rng.shuffle(rest)
        dev.update(chosen + rest[: max(0, want - len(chosen))])
    return sorted(dev)


def score(item_id, variant):
    qkey, render, model = VARIANTS[variant]
    item = ITEMS[item_id]
    questions = {f"g{i:02d}": {"type": "noul", "instructions": QUESTION[qkey].format(e=render(g))}
                 for i, g in enumerate(GROUPS)}
    r = httpx.post(URL, timeout=120, headers={"Authorization": "Bearer proxy-injected"},
                   json={"model": model, "state": {"user_message": item["query"]}, "questions": questions})
    r.raise_for_status()
    d = r.json()
    scores = {GROUPS[int(k[1:])]["name"]: v["noul"] for k, v in d["answers"].items()}
    return {"item": item_id, "variant": variant, "model": d.get("model"),
            "gold_group": ONTO["gold_filing"][item["gold_memory"]], "scores": scores,
            "input_tokens": d.get("usage", {}).get("input_tokens")}


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    dev = dev_split()
    (DIR / "dev_split.json").write_text(json.dumps(dev, indent=1))
    path = DIR / "scores.jsonl"
    done = {(json.loads(l)["item"], json.loads(l)["variant"]) for l in path.read_text().splitlines()} if path.exists() else set()
    jobs = [(i, v) for v in VARIANTS for i in dev if (i, v) not in done]
    print(f"{len(dev)} dev questions x {len(VARIANTS)} variants; {len(jobs)} Jev calls to run")

    def go(job):
        row = score(*job)
        with _lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    harness.pmap(go, jobs)
    rows = [json.loads(l) for l in path.read_text().splitlines() if json.loads(l)["item"] in set(dev)]
    summary = {}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        curve = []
        for c in CUTOFFS:
            recall = sum(r["scores"][r["gold_group"]] >= c for r in rs) / len(rs)
            groups = sum(sum(s >= c for s in r["scores"].values()) for r in rs) / len(rs)
            curve.append({"cutoff": c, "gold_group_recall": round(recall, 3), "groups_searched": round(groups, 2)})
        summary[v] = {"n": len(rs), "curve": curve,
                      "input_tokens_per_call": round(sum(r["input_tokens"] or 0 for r in rs) / len(rs))}
    (DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\ncutoff:            " + "  ".join(f"{c:>9}" for c in CUTOFFS))
    for v, s in summary.items():
        print(f"{v:24s}" + "  ".join(f"{p['gold_group_recall']:>4.0%}/{p['groups_searched']:<4.1f}" for p in s["curve"]))
    print("\n(each cell: share of questions where the gold group was chosen / average groups searched)")


if __name__ == "__main__":
    main()
