"""Run trials: build each setup's context, get Haiku's reply, grade it, critique it.

    uv run python -m build.run pilot --estimate
    uv run python -m build.run pilot

A trial is (question, setup, gold depth, shuffle). Setups:
  A        every memory in the system prompt
  E        shared hybrid search over all memories, top 10
  B1       every procedural memory (by ontology type) + search over the rest, top 10
  G        shared search inside every ontology group, top 10 each
  C1-C3    the model reads an ontology index (names / + descriptions / + applies when,
           examples, size, extras) and calls search_memory_group(group); each call
           returns that group's top 10 for the user's message
  C-embed  the 3 groups whose C3 entries are most similar to the message are searched
  C-Jev    Jev scores every group in one call; groups with p >= 0.5 are searched
  D        only the gold memory
  none     no memory at all (baseline)

Controls: the same user message, prompt template, store, shuffle and grading in
every setup. Memories carry opaque per-shuffle ids. Retrieved memories keep their
shuffle order (search rank never sets order) and the gold memory, when present,
is placed at the trial's depth: top, middle or end of the list the model sees.
Trials are logged to data/runs/<plan>.jsonl and are resumable.
"""

import argparse
import hashlib
import json
import random
import threading
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel

from . import config, harness, search
from .grade import grade

RUNS = config.DATA / "runs"
STORE = json.loads((config.DATA / "store_v1.json").read_text())
ONTO = json.loads((config.DATA / "ontology_v1.json").read_text())
GROUPS = ONTO["groups"]
GROUP_NAMES = [g["name"] for g in GROUPS]
GROUP_TYPE = {g["name"]: g["memory_type"] for g in GROUPS}
MEM = {m["id"]: m for m in STORE["memories"]}
ITEMS = {json.loads(l)["id"]: json.loads(l) for l in (config.DATA / "items_final.jsonl").read_text().splitlines()}
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_THRESHOLD = 0.1   # dev split (build/jev_tune.py): 99% gold-group recall, ~15 of 26 groups; chosen over 0.3 (91%, ~6) to favor recall
C_EMBED_GROUPS = 3
DEPTHS = {"top": 0.0, "middle": 0.5, "end": 1.0}
_lock = threading.Lock()

C_INSTRUCTION = (
    "You have a long-term memory about this user, organized into the groups in the index "
    "below. Before answering, read the index and decide which groups, if any, could matter "
    "for this request, including groups that apply to every reply. Search each of those "
    "groups with the search_memory_group tool (it searches with the user's message), then "
    "use the memories that are relevant. If no group could matter, answer directly.")
SEARCH_TOOL = {"name": "search_memory_group",
               "description": "Search one group of your long-term memory about the user. Returns "
                              "the group's most relevant memories for the user's message.",
               "input_schema": {"type": "object", "properties": {
                   "group": {"type": "string", "enum": GROUP_NAMES}}, "required": ["group"]}}


# --- memories, ids and placement -------------------------------------------------

def opaque(shuffle, key):
    return "m" + hashlib.sha1(f"{shuffle}:{key}".encode()).hexdigest()[:5]


def corpus(item, shuffle, version):
    """The store in this shuffle's order plus the gold (or twin) memory, with opaque ids,
    each tagged with its ontology group. The gold sits last here; placement comes later."""
    order = STORE["shuffles"][shuffle]
    secondary = set(item.get("secondary_gold") or [])
    if version == "twin":  # secondary golds would contradict the twin, so they sit out
        order = [k for k in order if k not in secondary]
    rows = [{"key": k, "id": opaque(shuffle, k), "memory": MEM[k]["memory"],
             "group": ONTO["store_filing"][k], "gold": False, "secondary": k in secondary} for k in order]
    text = item.get(f"{version}_memory")
    if text:
        rows.append({"key": "GOLD", "id": opaque(shuffle, "GOLD:" + item["id"]), "memory": text,
                     "group": ONTO["gold_filing"][text], "gold": True, "secondary": False})
    return rows


def place(rows, depth, position):
    """Keep non-gold rows in shuffle order (by `position`), put the gold row at depth."""
    others = sorted([r for r in rows if not r["gold"]], key=lambda r: position[r["key"]])
    gold = [r for r in rows if r["gold"]]
    if not gold:
        return others
    i = round(DEPTHS[depth] * len(others))
    return others[:i] + gold + others[i:]


def as_json(rows):
    return [{"id": r["id"], "memory": r["memory"]} for r in rows]


def gold_rank(query, rows):
    _, ranking = search.hybrid_search(query, rows, k=len(rows))
    ids = [x["id"] for x in ranking]
    g = next((r["id"] for r in rows if r["gold"]), None)
    return ids.index(g) + 1 if g in ids else None


# --- ontology renderings ----------------------------------------------------------

def entry(g, level):
    e = {"group": g["name"]}
    if level >= 2:
        e["description"] = g["description"]
    if level >= 3:
        e.update({"applies_when": g["applies_when"], "examples": g["examples"],
                  "size": g["size"], "extras": g["extras"]})
    return e


def entry_text(g):
    return (f"{g['name']}. {g['description']} Applies when: {g['applies_when']} "
            f"Examples: {' | '.join(g['examples'])}")


def jev_scores(query):
    # Tuned variant: reworded question, short entry (name + description).
    questions = {f"g{i:02d}": {"type": "noul", "instructions":
                 "Could any memory in this group change or improve the assistant's reply to user_message? "
                 "Groups that apply to every reply always count. "
                 f"Group entry: {g['name']}. {g['description']}"} for i, g in enumerate(GROUPS)}
    from .jev_memories import jev_post
    d = jev_post({"model": "jev-latest", "state": {"user_message": query}, "questions": questions})
    scores = {GROUPS[int(k[1:])]["name"]: v["noul"] for k, v in d["answers"].items()}
    return scores, d.get("model"), d.get("usage", {})


# --- the model call ---------------------------------------------------------------

def agent(system, question, tools, handler, max_rounds):
    """Haiku with tools: memory search handled by `handler`, mock task tools stubbed."""
    messages = [{"role": "user", "content": question}]
    trace, text, thinking, tin, tout, stop = [], "", "", 0, 0, None
    for _ in range(max_rounds):
        r = harness._create(config.ANSWER_MODEL, system, messages, None) if not tools else \
            harness.client().messages.create(model=config.ANSWER_MODEL, max_tokens=8000, system=system,
                                             thinking={"type": "adaptive", "display": "summarized"},
                                             output_config={"effort": "medium"}, messages=messages, tools=tools)
        tin += r.usage.input_tokens
        tout += r.usage.output_tokens
        stop = r.stop_reason
        text += "".join(b.text for b in r.content if b.type == "text")
        thinking += "\n".join(getattr(b, "thinking", "") or "" for b in r.content if b.type == "thinking")
        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:
            break
        results = []
        for b in calls:
            trace.append({"tool": b.name, "input": b.input})
            content = handler(b) if b.name == SEARCH_TOOL["name"] else harness.STUB_RESULT
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
        messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": results}]
    task_trace = [t["tool"] for t in trace if t["tool"] != SEARCH_TOOL["name"]]
    return {"reply": text, "thinking": thinking, "tool_trace": task_trace,
            "first_tool": task_trace[0] if task_trace else None,
            "memory_searches": [t["input"].get("group") for t in trace if t["tool"] == SEARCH_TOOL["name"]],
            "stop_reason": stop, "input_tokens": tin, "output_tokens": tout}


def system_with(rows):
    return harness.system_prompt(as_json(rows) if rows is not None else None)


def run_trial(t):
    item = ITEMS[t["item"]]
    q, setup, depth, shuffle, version = item["query"], t["setup"], t["depth"], t["shuffle"], t["version"]
    rows = corpus(item, shuffle, version)
    position = {r["key"]: i for i, r in enumerate(rows)}
    task_tools = None
    if item.get("tools"):
        task_tools = list(item["tools"])
        random.Random(f"{t['key']}").shuffle(task_tools)
    task_defs = harness.tool_defs(task_tools) if task_tools else []
    note = harness.TASK_NOTE if task_tools else ""
    info, extra_cost = {}, 0.0

    def plain(context_rows):
        system = system_with(context_rows) + note
        rounds = harness.MAX_TOOL_ROUNDS if task_tools else 1
        return agent(system, q, task_defs, None, rounds), context_rows

    if setup == "none":
        res, ctx = plain(None)
    elif setup == "D":
        res, ctx = plain([r for r in rows if r["gold"]])
    elif setup == "A":
        res, ctx = plain(place(rows, depth, position))
    elif setup == "E":
        top, _ = search.hybrid_search(q, rows)
        info["gold_rank"] = gold_rank(q, rows)
        res, ctx = plain(place(top, depth, position))
    elif setup == "B1":
        rules = [r for r in rows if GROUP_TYPE[r["group"]] == "procedural"]
        rest = [r for r in rows if GROUP_TYPE[r["group"]] != "procedural"]
        top, _ = search.hybrid_search(q, rest)
        gold_is_rule = any(r["gold"] for r in rules)
        rules_block = place(rules, depth, position) if gold_is_rule else place(rules, "top", position)
        top_block = place(top, depth, position) if not gold_is_rule else place(top, "top", position)
        res, ctx = plain(rules_block + top_block)
    elif setup in ("G", "C-embed", "C-Jev"):
        if setup == "G":
            chosen = GROUP_NAMES
        elif setup == "C-embed":
            gv = search.embed([entry_text(g) for g in GROUPS])
            sims = gv @ search.embed([q])[0]
            chosen = [GROUPS[i]["name"] for i in np.argsort(-sims)[:C_EMBED_GROUPS]]
            info["group_similarity"] = {GROUPS[i]["name"]: round(float(s), 4) for i, s in enumerate(sims)}
        else:
            scores, jmodel, jusage = jev_scores(q)
            chosen = [g for g, p in scores.items() if p >= JEV_THRESHOLD]
            info.update({"jev_scores": scores, "jev_model": jmodel, "jev_usage": jusage})
        found = []
        for g in chosen:
            top, _ = search.hybrid_search(q, [r for r in rows if r["group"] == g])
            found += top
        info["groups_searched"] = chosen
        res, ctx = plain(place(found, depth, position))
    elif setup in ("C1", "C2", "C3"):
        level = int(setup[1])
        index = [entry(g, level) for g in GROUPS]
        system = ("You are a helpful assistant.\n\n" + C_INSTRUCTION + "\n\nMemory index:\n"
                  + json.dumps(index, ensure_ascii=False, indent=1) + note)
        seen = []

        def handler(block):
            g = block.input.get("group")
            members = [r for r in rows if r["group"] == g]
            top, _ = search.hybrid_search(q, members)
            shown = place(top, depth, position)
            seen.extend(shown)
            return config.MEMORY_INSTRUCTION + "\n\n" + json.dumps(as_json(shown), ensure_ascii=False, indent=1)

        tools = [SEARCH_TOOL] + task_defs
        res = agent(system, q, tools, handler, max_rounds=6)
        ctx = seen
        info["groups_searched"] = res["memory_searches"]
    else:
        raise ValueError(setup)

    gold_row = next((r for r in rows if r["gold"]), None)
    in_ctx = [r for r in ctx or []]
    gold_in = gold_row is not None and any(r["gold"] for r in in_ctx)
    if gold_row is not None and "groups_searched" in info:
        info["gold_group"] = gold_row["group"]
        info["gold_group_searched"] = gold_row["group"] in info["groups_searched"]
    g = grade(item, res, version)
    secondary_in = sum(1 for r in in_ctx if r.get("secondary"))
    row = {**t, "scenario": item["scenario"], "gold_in_context": gold_in,
           "secondary_in_context": secondary_in, "any_gold_in_context": gold_in or secondary_in > 0,
           "n_secondary": len(item.get("secondary_gold") or []),
           "n_memories_in_context": len(in_ctx), **info, **res,
           "passes": g["passes"], "grade_method": g["method"], "judge_tokens": g["judge_tokens"]}
    if gold_in:
        row["critic"] = critic(item, res, gold_row["memory"])
    return row


# --- reasoning critic --------------------------------------------------------------

class Critique(BaseModel):
    label: Literal["not_mentioned", "mentioned_set_aside", "mentioned_applied"]
    reason: str


CRITIC_SYSTEM = """You review how an AI assistant used one stored memory about the user. You get \
the user's request, the memory, a summary of the assistant's reasoning, and its reply. The \
assistant may have had many other memories; judge ONLY the one memory you are given, and ignore \
whatever else the reply draws on. Label that memory's use as exactly one of:
- not_mentioned: neither the reasoning nor the reply shows the memory was considered.
- mentioned_set_aside: the memory was considered but not acted on.
- mentioned_applied: the memory was considered and shaped the reply.
Give a one-sentence reason."""


def critic(item, res, memory, model=config.JUDGE_MODEL):
    r = harness.client().messages.parse(
        model=model, max_tokens=3000, system=CRITIC_SYSTEM, output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"Request:\n{item['query']}\n\nMemory:\n{memory}\n\n"
                   f"Reasoning summary:\n{res['thinking'] or '(none)'}\n\nReply:\n{res['reply']}"}],
        output_format=Critique)
    u = r.usage
    return {"label": r.parsed_output.label, "reason": r.parsed_output.reason, "model": model,
            "tokens": [u.input_tokens, u.output_tokens]}


# --- plans -------------------------------------------------------------------------

SETUPS_ALL = ["A", "E", "B1", "G", "C1", "C2", "C3", "C-embed", "C-Jev"]


def pilot_items():
    by = {}
    for it in ITEMS.values():
        by.setdefault(it["scenario"], []).append(it)
    rng = random.Random(42)
    return [it["id"] for sc in sorted(by) for it in rng.sample(sorted(by[sc], key=lambda x: x["id"]), 5)]


PLANS = {
    "smoke": dict(items=lambda: pilot_items()[::7], setups=SETUPS_ALL, depths=["middle"], shuffles=[0]),
    "pilot": dict(items=pilot_items, setups=SETUPS_ALL, depths=["top", "middle"], shuffles=[0]),
}


def trials(plan):
    out = []
    for i in plan["items"]():
        it = ITEMS[i]
        for sh in plan["shuffles"]:
            out.append(dict(item=i, setup="none", depth="none", shuffle=sh, version="gold"))
            if it["gold_memory"]:
                out.append(dict(item=i, setup="D", depth="top", shuffle=sh, version="gold"))
            for s in plan["setups"]:
                for d in (plan["depths"] if it["gold_memory"] else ["none"]):
                    out.append(dict(item=i, setup=s, depth=d if d != "none" else "middle", shuffle=sh, version="gold"))
    for t in out:
        t["key"] = f"{t['item']}|{t['setup']}|{t['depth']}|{t['shuffle']}|{t['version']}"
    return out


def estimate(ts):
    """Rough $ from context sizes (Haiku, standard rate), before running."""
    per = {"none": 0.6, "D": 0.6, "A": 50, "E": 1.0, "B1": 7.0, "G": 13, "C1": 4, "C2": 5, "C3": 8,
           "C-embed": 2.0, "C-Jev": 2.0}  # thousand input tokens per trial, roughly
    answer = sum(per[t["setup"]] * 1000 * 0.10 / 1e6 + 900 * 0.50 / 1e6 for t in ts)
    return answer * 1.6  # judge, critic, C/tool rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", choices=PLANS)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ts = trials(PLANS[args.plan])
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{args.plan}.jsonl"
    done = {json.loads(l)["key"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [t for t in ts if t["key"] not in done]
    est = estimate(todo)
    print(f"{args.plan}: {len(ts)} trials, {len(todo)} to run, estimated ${est:.2f}")
    if args.estimate:
        return
    if est > args.max_cost:
        raise SystemExit(f"Estimate ${est:.2f} exceeds --max-cost ${args.max_cost:.2f}; not starting.")

    def go(t):
        try:
            row = run_trial(t)
        except Exception as e:  # log and keep going; failed trials rerun next time
            print(f"  failed {t['key']}: {type(e).__name__}: {str(e)[:200]}")
            return None
        with _lock, path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = [r for r in harness.pmap(go, todo, workers=args.workers) if r]
    print(f"ran {len(rows)} trials")


if __name__ == "__main__":
    main()
