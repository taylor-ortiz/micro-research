"""Exhaustive conflict screen: every gold memory against every store memory.

    uv run python -m build.screen_all

For each question, Haiku reads all 1,000 store memories (100 per call) next to the
question, its gold and twin memories and their pass checks, and labels any memory
that:
  supports_target       on its own would lead to a reply that passes the gold check
  supports_alternative  on its own would lead to a reply that passes the twin check
  contradicts_target    states something incompatible with the gold memory
Rules: supports_target -> secondary gold for that question; the other two labels ->
the memory is removed from the store and replaced by a screened candidate of the
same kind, in the same shuffle positions. Questions with more than 3 secondary
golds are dropped. Replacements are screened the same way until none is flagged.

Updates data/store_v1.json, data/items_final.jsonl (adds "secondary_gold"),
data/ontology_v1.json (files replacements) and data/frozen.meta.json.
Raw labels: data/screen_all/raw.jsonl (resumable).
"""

import json
import random
import threading
from typing import Literal

from pydantic import BaseModel

from . import config, harness, store as store_mod
from .conflicts import candidate_pool
from .ontology import file_batch

DIR = config.DATA / "screen_all"
RAW = DIR / "raw.jsonl"
BATCH = 100
MAX_SECONDARY = 3
_lock = threading.Lock()


class Flag(BaseModel):
    number: int
    label: Literal["supports_target", "supports_alternative", "contradicts_target"]


class Flags(BaseModel):
    flags: list[Flag]


SYSTEM = """You audit a memory store for an experiment on AI assistants. You get a user's \
request, a target memory about the user with the check a reply must pass to count as \
following it, an alternative version of that memory with its own check, and a numbered list \
of other memories. Flag only memories that, on their own, would matter for this request:
- supports_target: by itself it would lead a good assistant to a reply that passes the \
target's check (e.g. "usually rides the light rail to events" for "does not own a car").
- supports_alternative: by itself it would lead to a reply that passes the alternative's check.
- contradicts_target: it states something incompatible with the target memory.
Leave out everything else, including memories merely on the same topic. Most memories should \
be left out; return an empty list if none apply."""


def checks(it):
    c = it["check"]
    if c["type"] == "style":
        return f"The reply follows: {it['gold_memory']}", f"The reply follows: {it['twin_memory']}"
    if c["type"] == "tool":
        return (f"Of the tools {c['gold_tool']} and {c['twin_tool']}, the reply uses {c['gold_tool']} first",
                f"Of the tools {c['gold_tool']} and {c['twin_tool']}, the reply uses {c['twin_tool']} first")
    if c["type"] == "allergy":
        return (f"The reply recommends no dish or ingredient containing {c['gold_allergen']}",
                f"The reply recommends no dish or ingredient containing {c['twin_allergen']}")
    return c["gold_pass"], c["twin_pass"]


def screen(it, mems):
    gold_check, twin_check = checks(it)
    listing = "\n".join(f"{n + 1}. {m['memory']}" for n, m in enumerate(mems))
    r = harness.client().messages.parse(
        model=config.JUDGE_MODEL, max_tokens=4000, system=SYSTEM, output_config={"effort": "low"},
        messages=[{"role": "user", "content":
                   f"Request: {it['query']}\nTarget memory: {it['gold_memory']}\nTarget's check: {gold_check}\n"
                   f"Alternative memory: {it['twin_memory']}\nAlternative's check: {twin_check}\n\n"
                   f"Other memories:\n{listing}"}],
        output_format=Flags)
    u = r.usage
    flags = [{"id": mems[f.number - 1]["id"], "label": f.label}
             for f in r.parsed_output.flags if 1 <= f.number <= len(mems)]
    return flags, (u.input_tokens * 0.10 + u.output_tokens * 0.50) / 1e6


def run_screen(items, mems, tag):
    """Screen every item against `mems`; returns {item_id: [flags]} and cost. Resumable."""
    DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            row = json.loads(l)
            done[(row["tag"], row["item"], row["batch"])] = row
    mems = sorted(mems, key=lambda m: m["id"])
    batches = [mems[i:i + BATCH] for i in range(0, len(mems), BATCH)]
    jobs = [(it, b) for it in items for b in range(len(batches)) if (tag, it["id"], b) not in done]
    print(f"[{tag}] {len(jobs)} screen calls to run ({len(items)} items x {len(batches)} batches)")

    def go(job):
        it, b = job
        flags, cost = screen(it, batches[b])
        row = {"tag": tag, "item": it["id"], "batch": b, "flags": flags, "cost": cost}
        with _lock, RAW.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    for row in harness.pmap(go, jobs):
        done[(row["tag"], row["item"], row["batch"])] = row
    out, cost = {}, 0.0
    for (t, i, _), row in done.items():
        if t == tag:
            out.setdefault(i, []).extend(row["flags"])
            cost += row["cost"]
    return out, cost


def main():
    data = json.loads((config.DATA / "store_v1.json").read_text())
    store = data["memories"]
    items = [json.loads(l) for l in (config.DATA / "items_final.jsonl").read_text().splitlines()]
    gold_items = [it for it in items if it["gold_memory"]]
    onto = json.loads((config.DATA / "ontology_v1.json").read_text())

    flags, total = run_screen(gold_items, store, "store")
    removed = {f["id"] for fl in flags.values() for f in fl if f["label"] != "supports_target"}
    secondary = {i: sorted({f["id"] for f in fl if f["label"] == "supports_target"} - removed)
                 for i, fl in flags.items()}

    # Replace removed memories with screened candidates of the same source, in place.
    used = {m["id"] for m in store} | set(json.loads((config.DATA / "conflicts.json").read_text())["dropped"])
    pool = [p for p in candidate_pool() if p["id"] not in used]
    rng = random.Random(99)
    rng.shuffle(pool)
    replacement, round_n = {}, 0
    pending = sorted(removed)
    src = {m["id"]: m["source"] for m in store}
    while pending:
        round_n += 1
        picks = []
        for rid in pending:
            cand = next((p for p in pool if p["source"] == src[rid] and p["id"] not in replacement.values()
                         and p["id"] not in picks), None)
            if cand is None:
                raise SystemExit(f"No replacement left for source {src[rid]}")
            picks.append(cand["id"])
            replacement[rid] = cand["id"]
            pool.remove(cand)
        cand_mems = [p for p in candidate_pool() if p["id"] in set(picks)]
        cflags, c = run_screen(gold_items, cand_mems, f"replace{round_n}")
        total += c
        bad = {f["id"] for fl in cflags.values() for f in fl}
        pending = sorted(rid for rid, cid in replacement.items() if cid in bad)
        for rid in pending:
            src[rid] = src[rid]
        print(f"round {round_n}: {len(picks)} replacements screened, {len(pending)} flagged again")

    new_by_id = {p["id"]: p for p in candidate_pool()}
    final_ids = {rid: replacement[rid] for rid in removed}
    data["memories"] = sorted([m for m in store if m["id"] not in removed]
                              + [new_by_id[cid] for cid in final_ids.values()], key=lambda m: m["id"])
    data["shuffles"] = [[final_ids.get(k, k) for k in order] for order in data["shuffles"]]
    (config.DATA / "store_v1.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))

    # File replacements into the ontology with the same process as everything else.
    groups = onto["groups"]
    new_mems = [new_by_id[c] for c in final_ids.values()]
    for i in range(0, len(new_mems), 25):
        chunk = new_mems[i:i + 25]
        picks, c = file_batch(groups, [m["memory"] for m in chunk])
        total += c
        for m, g in zip(chunk, picks):
            onto["store_filing"][m["id"]] = g
    for rid in removed:
        onto["store_filing"].pop(rid, None)
    texts = {m["id"]: m["memory"] for m in data["memories"]}
    removed_texts = {m["memory"] for m in store if m["id"] in removed}
    rng2 = random.Random(11)
    for g in groups:
        members = [k for k, gg in onto["store_filing"].items() if gg == g["name"]]
        g["size"] = len(members)
        keep = [e for e in g["examples"] if e not in removed_texts]
        extra = [texts[k] for k in rng2.sample(members, min(3, len(members))) if texts[k] not in keep]
        g["examples"] = (keep + extra)[:3]
    (config.DATA / "ontology_v1.json").write_text(json.dumps(onto, ensure_ascii=False, indent=1))

    kept, dropped_q = [], []
    for it in items:
        sec = secondary.get(it["id"], []) if it["gold_memory"] else []
        if len(sec) > MAX_SECONDARY:
            dropped_q.append(it["id"])
            continue
        it["secondary_gold"] = sec
        kept.append(it)
    (config.DATA / "items_final.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in kept))

    meta = json.loads((config.DATA / "frozen.meta.json").read_text())
    meta.update({"store_v1.json": store_mod.sha(config.DATA / "store_v1.json"),
                 "items_final.jsonl": store_mod.sha(config.DATA / "items_final.jsonl"),
                 "ontology_v1.json": store_mod.sha(config.DATA / "ontology_v1.json"),
                 "exhaustive_screen": {"removed": len(removed), "questions_dropped": dropped_q,
                                       "questions_with_secondary": sum(1 for it in kept if it.get("secondary_gold")),
                                       "cost": round(total, 3)}})
    counts = {}
    for it in kept:
        counts[it["scenario"]] = counts.get(it["scenario"], 0) + 1
    meta["item_counts"] = counts
    (config.DATA / "frozen.meta.json").write_text(json.dumps(meta, indent=1))
    (DIR / "results.json").write_text(json.dumps({"removed": sorted(removed), "replacement": final_ids,
                                                  "secondary": secondary, "dropped_questions": dropped_q}, indent=1))
    print(json.dumps(meta["exhaustive_screen"], indent=1))
    print("items:", counts)


if __name__ == "__main__":
    main()
