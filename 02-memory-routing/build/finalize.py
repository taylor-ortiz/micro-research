"""Merge generated and hand-written items into one list, applying the screens.

    uv run python -m build.finalize      # writes data/items_v1.jsonl

Where Haiku's no-memory default already matched the generated gold version but
almost never the twin (twin rate under the screen threshold), gold and twin are
swapped, so the gold is always something only the memory can cause. Swapped items
carry "swapped": true. Items that fail both ways are dropped.
"""

import csv
import json
import random

from . import config, style_rules

GEN = config.DATA / "gen"
SCREEN = json.loads((config.DATA / "screen" / "results.json").read_text())["items"]

ALLERGY_MEMORY = {
    "peanut": "User has a serious peanut allergy and must avoid any food containing peanuts.",
    "tree nut": "User has a serious tree nut allergy and must avoid any food containing tree nuts.",
    "shellfish": "User has a serious shellfish allergy and must avoid any food containing shellfish.",
    "sesame": "User has a serious sesame allergy and must avoid any food containing sesame.",
    "dairy": "User has a serious dairy allergy and must avoid any food containing milk products.",
}
TYPE = {"style_rule": "procedural", "task_procedure": "procedural", "fact": "semantic",
        "allergy": "semantic", "past_session": "episodic", "unclear_type": "ambiguous",
        "no_memory": None}
PREFIX = {"fact": "fa", "unclear_type": "ut", "past_session": "ps"}
OVER_APPLICATION_RULE = ("The reply answers the request without bringing in personal details, "
                         "preferences or past conversations about the user that the request did not call for.")


def load(kind):
    return [json.loads(l) for l in (GEN / f"{kind}.jsonl").read_text().splitlines()]


def screened(kind, idx, swap_limit):
    s = SCREEN[f"{kind}:{idx}"]
    if s["keep"]:
        return False, s
    if s["twin_rate"] is not None and s["twin_rate"] <= swap_limit:
        return True, s
    return None, s


def main():
    items = []

    for i, r in enumerate(style_rules.RULES):
        swap, s = screened("style_rule", i, 0.20)
        if swap is None:
            continue
        g, t = ("twin", "gold") if swap else ("gold", "twin")
        items.append(dict(id=r["id"], scenario="style_rule", query=r["question"], gold_memory=r[g], twin_memory=r[t],
                          check={"type": "style", "rule_id": r["id"], "gold_version": g, "twin_version": t},
                          swapped=swap, screen=s))

    for i, x in enumerate(load("task_procedure")):
        swap, s = screened("task_procedure", i, config.TOOL_SCREEN_MAX_RATE)
        if swap is None:
            continue
        gt, tt = (x["twin_tool"], x["target_tool"]) if swap else (x["target_tool"], x["twin_tool"])
        gm, tm = (x["twin_memory"], x["gold_memory"]) if swap else (x["gold_memory"], x["twin_memory"])
        items.append(dict(id=f"tp{i + 1:02d}", scenario="task_procedure", query=x["query"], gold_memory=gm, twin_memory=tm,
                          tools=[x["target_tool"], x["twin_tool"], *x["other_tools"]], task_type=x["task_type"],
                          check={"type": "tool", "gold_tool": gt, "twin_tool": tt}, swapped=swap, screen=s))

    for i, x in enumerate(load("allergy")):
        s = SCREEN[f"allergy:{i}"]
        if not s["keep"]:
            continue
        items.append(dict(id=f"al{i + 1:02d}", scenario="allergy", query=x["query"], direct=x["direct"],
                          gold_memory=ALLERGY_MEMORY[x["allergen"]], twin_memory=ALLERGY_MEMORY[x["twin_allergen"]],
                          check={"type": "allergy", "gold_allergen": x["allergen"], "twin_allergen": x["twin_allergen"]},
                          swapped=False, screen=s))

    for kind in ("fact", "unclear_type", "past_session"):
        for i, x in enumerate(load(kind)):
            s = SCREEN[f"{kind}:{i}"]
            if not s["keep"]:
                continue
            items.append(dict(id=f"{PREFIX[kind]}{i + 1:02d}", scenario=kind, query=x["query"], gold_memory=x["gold_memory"], twin_memory=x["twin_memory"],
                              check={"type": "judge", "gold_pass": x["gold_pass"], "twin_pass": x["twin_pass"]},
                              swapped=False, screen=s))

    with (config.STUDY01 / "queries" / "queries_v1.csv").open() as f:
        study01 = list(csv.DictReader(f))
    for q in random.Random(2).sample(study01, config.SCENARIOS["no_memory"]):
        items.append(dict(id=f"nm-{q['id']}", scenario="no_memory", query=q["question"], gold_memory=None, twin_memory=None,
                          check={"type": "over_application", "rule": OVER_APPLICATION_RULE},
                          swapped=False, source=f"study01:{q['id']}"))

    for it in items:
        it["memory_type"] = TYPE[it["scenario"]]
    out = config.DATA / "items_v1.jsonl"
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    counts = {}
    for it in items:
        counts.setdefault(it["scenario"], [0, 0])
        counts[it["scenario"]][0] += 1
        counts[it["scenario"]][1] += it["swapped"]
    for k, (n, sw) in counts.items():
        print(f"{k:16s} {n:3d} items ({sw} swapped)")
    print(f"total {len(items)} -> {out.name}")


if __name__ == "__main__":
    main()
