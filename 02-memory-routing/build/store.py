"""Assemble and freeze the 1,000-memory distractor store and the final item set.

    uv run python -m build.store

- Distractors: 600 semantic (haystack_v1 everyday facts), 250 episodic (generated
  past sessions), 150 procedural (task decoys, 40 generated every-reply rules, and
  haystack_v1 content-choice rules for the rest), all drawn from what survived the
  conflict screen. Gold memories are added
  per trial, so they are not in this file.
- Items: those that passed both the no-memory screen and perfect-retrieval validation.
- Shuffles: 5 fixed orders of the store, shared by every condition and depth.

Writes data/store_v1.json, data/items_final.jsonl and data/frozen.meta.json (SHA-256 of each).
"""

import hashlib
import json
import random

from . import config
from .conflicts import candidate_pool

N_SHUFFLES = 5
SEED = 2026


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dropped = set(json.loads((config.DATA / "conflicts.json").read_text())["dropped"])
    pool = [p for p in candidate_pool() if p["id"] not in dropped]
    rng = random.Random(SEED)

    def take(rows, n, label):
        if len(rows) < n:
            raise SystemExit(f"Not enough {label}: {len(rows)} < {n}")
        return rng.sample(rows, n)

    semantic = take([p for p in pool if p["type"] == "semantic"], config.N_SEMANTIC, "semantic")
    episodic = take([p for p in pool if p["type"] == "episodic"], config.N_EPISODIC, "episodic")
    decoys = [p for p in pool if p["source"] == "task_decoy"]
    n_decoys = min(config.N_TASK_DECOYS, len(decoys))
    universal = take([p for p in pool if p["source"] == "universal_rule"], config.N_UNIVERSAL_RULES,
                     "every-reply rules")
    rules = take([p for p in pool if p["source"] == "haystack_v1:content_choices"],
                 config.N_PROCEDURAL - n_decoys - config.N_UNIVERSAL_RULES, "content-choice rules")
    store = semantic + episodic + rng.sample(decoys, n_decoys) + universal + rules
    assert len(store) == config.STORE_SIZE and len({s["id"] for s in store}) == len(store)

    orders = []
    for k in range(N_SHUFFLES):
        ids = [s["id"] for s in store]
        random.Random(SEED + k + 1).shuffle(ids)
        orders.append(ids)
    store_path = config.DATA / "store_v1.json"
    store_path.write_text(json.dumps({"memories": sorted(store, key=lambda s: s["id"]), "shuffles": orders},
                                     ensure_ascii=False, indent=1))

    items = [json.loads(l) for l in (config.DATA / "items_v1.jsonl").read_text().splitlines()]
    valid = json.loads((config.DATA / "validate" / "results.json").read_text())["items"]
    final = [it for it in items if it["gold_memory"] is None or valid.get(it["id"], {}).get("keep")]
    items_path = config.DATA / "items_final.jsonl"
    items_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in final))

    counts = {}
    for it in final:
        counts[it["scenario"]] = counts.get(it["scenario"], 0) + 1
    meta = {"store_v1.json": sha(store_path), "items_final.jsonl": sha(items_path),
            "store_counts": {t: sum(s["type"] == t for s in store) for t in ("semantic", "episodic", "procedural")},
            "task_decoys": n_decoys, "every_reply_rules": config.N_UNIVERSAL_RULES, "item_counts": counts, "shuffles": N_SHUFFLES, "seed": SEED}
    (config.DATA / "frozen.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
