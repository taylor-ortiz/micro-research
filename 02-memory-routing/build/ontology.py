"""Build the memory ontology used by G, C1-C3, C-embed and C-Jev (and B1's types).

    uv run python -m build.ontology

1. Opus reads the whole distractor store (no questions) and proposes 20-25 groups
   under four types, writing each group's entry: name, description, applies when,
   and type-specific extras.
2. Haiku files every memory into one group, 25 at a time, seeing only the group
   names and descriptions. Gold and twin memories are filed by exactly the same
   process, as new memories would be.
3. Examples (3 per group) are drawn from the group's distractors, and size counts
   them. A memory's type, for B1, is its group's type.

Writes data/ontology_v1.json.
"""

import json
import random
from typing import Literal

from pydantic import BaseModel

from . import config, harness

TYPES = ["style rules", "task procedures", "facts", "past sessions"]
TYPE_OF = {"style rules": "procedural", "task procedures": "procedural",
           "facts": "semantic", "past sessions": "episodic"}
OUT = config.DATA / "ontology_v1.json"


class Group(BaseModel):
    name: str
    type: Literal["style rules", "task procedures", "facts", "past sessions"]
    description: str
    applies_when: str
    extras: str


class Proposal(BaseModel):
    groups: list[Group]


PROPOSE = """You design the index for an AI assistant's long-term memory about one user. \
You will see every memory in the store. Group them into 20 to 25 topic groups, each under \
exactly one of four types: style rules (how the user wants replies written), task procedures \
(how the user wants tasks done with tools), facts (facts about the user's life), past sessions \
(dated summaries of earlier conversations).

For each group write:
- name: "<Type>: <topic>", e.g. "Facts: health and fitness".
- description: one sentence on what the memories in it cover.
- applies_when: one sentence saying when an assistant should consult this group before \
replying, judged by what the memories are for, not by topic words alone (e.g. style rules \
that govern every reply apply to every reply).
- extras: for task procedures, the tools the memories name; for past sessions, the kinds of \
projects and the date range; otherwise a few typical keywords.

Groups must not overlap, should be of broadly similar size (no group over 120 memories), and \
should leave room for new memories of the same kinds."""


def propose(store):
    listing = "\n".join(f"- {m['memory']}" for m in store)
    r = harness.client().messages.parse(
        model=config.GENERATOR_MODEL, max_tokens=16000, system=PROPOSE,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": f"The memory store ({len(store)} memories):\n{listing}"}],
        output_format=Proposal)
    u = r.usage
    return [g.model_dump() for g in r.parsed_output.groups], (u.input_tokens * 4 + u.output_tokens * 20) / 1e6


def file_batch(groups, memories, attempt=1):
    names = [g["name"] for g in groups]
    Choice = Literal[tuple(names)]  # type: ignore[valid-type]
    Filing = type("Filing", (BaseModel,), {"__annotations__": {"number": int, "group": Choice}})
    Batch = type("FilingBatch", (BaseModel,), {"__annotations__": {"filings": list[Filing]}})
    index = "\n".join(f"- {g['name']}: {g['description']}" for g in groups)
    listing = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(memories))
    r = harness.client().messages.parse(
        model=config.JUDGE_MODEL, max_tokens=8000, output_config={"effort": "low"},
        system="You file new memories into an assistant's memory index. For each numbered memory, "
               "give its number and the one group it belongs in. Return exactly one filing per memory.",
        messages=[{"role": "user", "content": f"Groups:\n{index}\n\nMemories:\n{listing}"}],
        output_format=Batch)
    by_n = {f.number: f.group for f in r.parsed_output.filings}
    u = r.usage
    cost = (u.input_tokens * 0.10 + u.output_tokens * 0.50) / 1e6
    if not all(i + 1 in by_n for i in range(len(memories))):
        if attempt >= 3:
            raise RuntimeError(f"Could not file batch after {attempt} attempts")
        picks, c = file_batch(groups, memories, attempt + 1)
        return picks, cost + c
    return [by_n[i + 1] for i in range(len(memories))], cost


def main():
    store = json.loads((config.DATA / "store_v1.json").read_text())["memories"]
    items = [json.loads(l) for l in (config.DATA / "items_final.jsonl").read_text().splitlines()]
    gold_texts = sorted({t for it in items for t in (it["gold_memory"], it["twin_memory"]) if t})

    # The proposer sees the store in a fixed random order, so no shuffle used in trials leaks in.
    shown = list(store)
    random.Random(7).shuffle(shown)
    cached = config.DATA / "ontology_proposal.json"
    if cached.exists():
        groups, cost = json.loads(cached.read_text()), 0.0
        print(f"using cached proposal: {len(groups)} groups")
    else:
        groups, cost = propose(shown)
        cached.write_text(json.dumps(groups, indent=1, ensure_ascii=False))
        print(f"proposed {len(groups)} groups (${cost:.2f})")

    texts = [m["memory"] for m in store] + gold_texts
    filed, fcost = {}, 0.0
    batches = [texts[i:i + 25] for i in range(0, len(texts), 25)]
    for batch, (picks, c) in zip(batches, harness.pmap(lambda b: file_batch(groups, b), batches)):
        filed.update(zip(batch, picks))
        fcost += c
    print(f"filed {len(filed)} memories (${fcost:.2f})")

    by_group = {g["name"]: [m for m in store if filed[m["memory"]] == g["name"]] for g in groups}
    rng = random.Random(11)
    for g in groups:
        members = by_group[g["name"]]
        g["size"] = len(members)
        g["examples"] = [m["memory"] for m in rng.sample(members, min(3, len(members)))]
        g["memory_type"] = TYPE_OF[g["type"]]
    groups = [g for g in groups if g["size"] > 0 or any(filed[t] == g["name"] for t in gold_texts)]

    OUT.write_text(json.dumps({"groups": groups,
                               "store_filing": {m["id"]: filed[m["memory"]] for m in store},
                               "gold_filing": {t: filed[t] for t in gold_texts},
                               "cost": round(cost + fcost, 4)}, indent=1, ensure_ascii=False))
    for g in sorted(groups, key=lambda g: g["name"]):
        print(f"  {g['size']:4d}  {g['name']}")


if __name__ == "__main__":
    main()
