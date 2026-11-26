"""Conflict screen: drop distractors that clash with any gold memory or its twin.

    uv run python -m build.conflicts

For each validated item, the 30 distractor candidates most similar (TF-IDF) to
its gold or twin memory go to Haiku, which flags any that contradict, duplicate
or would also change the answer to that item's question. Flagged distractors
are removed from the pool for every trial. Output: data/conflicts.json.
"""

import json

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, harness

TOP_K = 30


def candidate_pool():
    """All distractors that could go into the store, with a stable id and source."""
    hay = json.loads((config.STUDY01 / "data" / "haystack_v1.json").read_text())
    meta = {m["id"]: m for m in json.loads((config.STUDY01 / "data" / "haystack_v1.meta.json").read_text())["items"]}
    pool = []
    for h in hay:
        cat = meta[h["id"]]["category"]
        if cat in config.EXCLUDED_HAYSTACK_CATEGORIES:
            continue
        kind = "procedural" if meta[h["id"]]["kind"] == "output_neutral" else "semantic"
        pool.append({"id": f"h-{h['id']}", "memory": h["preference"], "type": kind, "source": f"haystack_v1:{cat}"})
    for kind, typ, pre in (("session_distractor", "episodic", "s"), ("task_decoy", "procedural", "d")):
        for i, l in enumerate((config.DATA / "gen" / f"{kind}.jsonl").read_text().splitlines()):
            pool.append({"id": f"{pre}-{i + 1:03d}", "memory": json.loads(l)["memory"], "type": typ, "source": kind})
    return pool


class Flags(BaseModel):
    conflicting_numbers: list[int]


SYSTEM = """You check a memory store for an experiment. You get a user's question, a target \
memory, an alternative version of it, and a numbered list of other memories. List the numbers \
of any other memory that (a) contradicts either version, (b) says the same thing as either \
version, or (c) would by itself change how an assistant answers the question in the way either \
version does. Unrelated memories, or ones on the same topic that would not change the answer, \
are fine. Return an empty list if none apply."""


def main():
    items = [json.loads(l) for l in (config.DATA / "items_v1.jsonl").read_text().splitlines()]
    valid = json.loads((config.DATA / "validate" / "results.json").read_text())["items"]
    items = [it for it in items if it["gold_memory"] and valid.get(it["id"], {}).get("keep")]
    pool = candidate_pool()
    texts = [p["memory"] for p in pool]
    vec = TfidfVectorizer(stop_words="english").fit(texts + [it["gold_memory"] for it in items])
    P = vec.transform(texts)

    def check(it):
        q = vec.transform([it["gold_memory"] + " " + it["twin_memory"] + " " + it["query"]])
        top = cosine_similarity(q, P)[0].argsort()[::-1][:TOP_K]
        listing = "\n".join(f"{n + 1}. {texts[i]}" for n, i in enumerate(top))
        r = harness.client().messages.parse(
            model=config.JUDGE_MODEL, max_tokens=4000, system=SYSTEM, output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"Question: {it['query']}\nTarget memory: {it['gold_memory']}\n"
                       f"Alternative version: {it['twin_memory']}\n\nOther memories:\n{listing}"}],
            output_format=Flags)
        flagged = [pool[top[n - 1]]["id"] for n in r.parsed_output.conflicting_numbers if 1 <= n <= len(top)]
        return it["id"], flagged, r.usage.input_tokens, r.usage.output_tokens

    out = harness.pmap(check, items)
    dropped = {}
    for item_id, flagged, _, _ in out:
        for pid in flagged:
            dropped.setdefault(pid, []).append(item_id)
    cost = sum(a * 0.10 + b * 0.50 for _, _, a, b in out) / 1e6
    by_type = {}
    for p in pool:
        by_type.setdefault(p["type"], [0, 0])
        by_type[p["type"]][0] += 1
        by_type[p["type"]][1] += p["id"] in dropped
    (config.DATA / "conflicts.json").write_text(json.dumps({"dropped": dropped}, indent=1))
    for t, (n, d) in by_type.items():
        print(f"{t:11s} pool {n:5d}  dropped {d:4d}")
    print(f"items checked {len(items)}, cost ${cost:.2f}")


if __name__ == "__main__":
    main()
