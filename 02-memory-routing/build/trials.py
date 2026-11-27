"""Main runs: Haiku main, Haiku swap test, Sonnet side by side.

    uv run python -m build.trials main_haiku --estimate
    uv run python -m build.trials main_haiku --batch
    uv run python -m build.trials main_sonnet --batch

Same setups, controls and grading as build/run.py (the pilot), plus:
- the answering model is a trial setting (Haiku 5.5 or Sonnet 5.5, same effort and template);
- C-types: the model reads the 4 type entries and searches whole types;
- the 75 dev questions used to tune Jev are excluded; questions are split across the
  5 shuffles by a hash of their id;
- prompt caching: the system prompt is split where it first differs between questions
  (at the gold memory, or after B1's rules block), and the shared prefix is cached;
- --batch runs every model call through the Message Batches API (50% off), one round
  per batch, handling memory searches and stub tool results locally between rounds.
Judging and the critic run as standard calls afterwards. Trials are logged to
data/runs/<plan>.jsonl and are resumable.
"""

import argparse
import hashlib
import json
import random
import threading
import time

import numpy as np

from . import config, harness, search
from .grade import grade
from .run import (C_INSTRUCTION, GROUPS, GROUP_NAMES, GROUP_TYPE, ITEMS, ONTO, SEARCH_TOOL, JEV_THRESHOLD,
                  C_EMBED_GROUPS, as_json, corpus, critic, entry, entry_text, gold_rank, jev_scores, place)

RUNS = config.DATA / "runs"
TYPES = json.loads((config.DATA / "ontology_types.json").read_text())
TYPE_OF_GROUP = {g: t["name"] for t in TYPES for g in t["groups"]}
TYPE_TOOL = {**SEARCH_TOOL, "description": "Search one type of your long-term memory about the user. Returns the "
             "type's most relevant memories for the user's message.",
             "input_schema": {"type": "object", "properties": {"group": {"type": "string", "enum": [t["name"] for t in TYPES]}},
                              "required": ["group"]}}
JEVPLUS_KEEP = 0.7   # memory-score cutoff, chosen on the dev split (build/jev_memories.py)
JEVPLUS_CAP = 20
LPICK_KEEP = 0.6     # LLM-picker cutoff, chosen on the dev split by the same rule (build/llm_pick.py)
DEV = set(json.loads((config.DATA / "jev_tune" / "dev_split.json").read_text()))
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
# $ per million tokens (input, output); cache reads at 10% of input, writes at 125%.
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
_lock = threading.Lock()


# --- preparing a trial ---------------------------------------------------------------

def shuffle_of(item_id):
    """Questions are split across the 5 fixed shuffles by a hash of their id."""
    return int(hashlib.sha1(item_id.encode()).hexdigest(), 16) % 5


def render(rows):
    return "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in as_json(rows)) + "\n]"


def system_blocks(rows, split, note):
    """System prompt as text blocks. Rows before `split` are identical across questions in
    this shuffle and setup, so that prefix is marked for caching."""
    head = "You are a helpful assistant."
    if rows is None:
        return [{"type": "text", "text": head + note}]
    body = render(rows)
    full = head + "\n\n" + config.MEMORY_INSTRUCTION + "\n\n"
    if split and split > 0:
        js = as_json(rows)
        prefix = full + "[\n" + "".join(json.dumps(r, ensure_ascii=False) + ",\n" for r in js[:split])
        suffix = "".join(json.dumps(r, ensure_ascii=False) + (",\n" if i < len(js) - 1 else "\n")
                         for i, r in enumerate(js[split:], start=split)) + "]" + note
        return [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": suffix}]
    return [{"type": "text", "text": full + body + note}]


def summarize_ctx(ctx):
    return {"gold_in_context": any(r["gold"] for r in ctx),
            "secondary_in_context": sum(1 for r in ctx if r.get("secondary")),
            "n_memories_in_context": len(ctx)}


def prepare(t):
    item = ITEMS[t["item"]]
    q, setup, depth, shuffle, version = item["query"], t["setup"], t["depth"], t["shuffle"], t["version"]
    rows = corpus(item, shuffle, version)
    position = {r["key"]: i for i, r in enumerate(rows)}
    task_tools = None
    if item.get("tools"):
        task_tools = list(item["tools"])
        random.Random(t["key"].replace(f"|{t['model']}", "")).shuffle(task_tools)
    task_defs = harness.tool_defs(task_tools) if task_tools else []
    note = harness.TASK_NOTE if task_tools else ""
    st = {"t": t, "messages": [{"role": "user", "content": q}], "tools": task_defs or None,
          "max_rounds": harness.MAX_TOOL_ROUNDS if task_tools else 1, "round": 0, "done": False,
          "trace": [], "text": "", "thinking": "", "usage": [], "info": {}, "ctx": None, "stop": None}

    def gold_index(ctx):
        return next((i for i, r in enumerate(ctx) if r["gold"]), None)

    if setup == "none":
        st["system"], ctx = system_blocks(None, None, note), []
    elif setup == "D":
        ctx = [r for r in rows if r["gold"]]
        st["system"] = system_blocks(ctx, None, note)
    elif setup == "A":
        ctx = place(rows, depth, position)
        st["system"] = system_blocks(ctx, gold_index(ctx), note)
    elif setup == "E":
        top, _ = search.hybrid_search(q, rows)
        st["info"]["gold_rank"] = gold_rank(q, rows)
        ctx = place(top, depth, position)
        st["system"] = system_blocks(ctx, None, note)
    elif setup == "B1":
        rules = [r for r in rows if GROUP_TYPE[r["group"]] == "procedural"]
        rest = [r for r in rows if GROUP_TYPE[r["group"]] != "procedural"]
        top, _ = search.hybrid_search(q, rest)
        gold_is_rule = any(r["gold"] for r in rules)
        rules_block = place(rules, depth if gold_is_rule else "top", position)
        ctx = rules_block + place(top, "top" if gold_is_rule else depth, position)
        gi = gold_index(rules_block)
        st["system"] = system_blocks(ctx, gi if gold_is_rule else len(rules_block), note)
    elif setup in ("G", "C-embed", "C-Jev"):
        if setup == "G":
            chosen = GROUP_NAMES
        elif setup == "C-embed":
            sims = search.embed([entry_text(g) for g in GROUPS]) @ search.embed([q])[0]
            chosen = [GROUPS[i]["name"] for i in np.argsort(-sims)[:C_EMBED_GROUPS]]
            st["info"]["group_similarity"] = {GROUPS[i]["name"]: round(float(s), 4) for i, s in enumerate(sims)}
        else:
            scores, jmodel, jusage = jev_scores(q)
            chosen = [g for g, p in scores.items() if p >= JEV_THRESHOLD]
            st["info"].update({"jev_scores": scores, "jev_model": jmodel, "jev_usage": jusage})
        found = []
        for g in chosen:
            top, _ = search.hybrid_search(q, [r for r in rows if r["group"] == g])
            found += top
        st["info"]["groups_searched"] = chosen
        ctx = place(found, depth, position)
        st["system"] = system_blocks(ctx, None, note)
    elif setup == "C-Jev+":
        # Jev picks groups (cutoff 0.1), then scores every memory in them; memories at
        # JEVPLUS_KEEP or above go in (at most JEVPLUS_CAP, highest first), in shuffle order.
        from .jev_memories import score_memories
        scores, jmodel, _ = jev_scores(q)
        chosen = [g for g, p in scores.items() if p >= JEV_THRESHOLD]
        cands = [r for r in rows if r["group"] in chosen]
        mscores, jtok = score_memories(q, cands)
        kept = sorted([r for r in cands if mscores[r["id"]] >= JEVPLUS_KEEP], key=lambda r: -mscores[r["id"]])[:JEVPLUS_CAP]
        gold = next((r for r in cands if r["gold"]), None)
        st["info"].update({"groups_searched": chosen, "jev_scores": scores, "jev_model": jmodel,
                           "n_candidates": len(cands), "n_kept": len(kept), "jev_memory_tokens": jtok,
                           "gold_memory_score": mscores.get(gold["id"]) if gold else None,
                           "gold_memory_jev_rank": (sorted(cands, key=lambda r: -mscores[r["id"]]).index(gold) + 1) if gold else None})
        ctx = place(kept, depth, position)
        st["system"] = system_blocks(ctx, None, note)
    elif setup in ("L-cat", "L-all"):
        # Haiku picks the memories (build/llm_pick.py): L-cat from Jev's groups, L-all from
        # the whole store. Same keep rule as C-Jev+, with the cutoff tuned the same way.
        from .llm_pick import cached_pick
        pk = cached_pick(setup, item, shuffle, version)
        cands = [r for r in rows if r["key"] in pk["scores"]]
        kept = sorted([r for r in cands if pk["scores"][r["key"]] >= LPICK_KEEP],
                      key=lambda r: -pk["scores"][r["key"]])[:JEVPLUS_CAP]
        st["info"].update({"n_candidates": len(cands), "n_kept": len(kept),
                           "gold_memory_score": pk["scores"].get("GOLD"),
                           "picker_input_tokens": pk["picker_input_tokens"],
                           "picker_output_tokens": pk["picker_output_tokens"],
                           "picker_seconds": pk["picker_seconds"]})
        if pk.get("groups") is not None:
            st["info"]["groups_searched"] = pk["groups"]
        ctx = place(kept, depth, position)
        st["system"] = system_blocks(ctx, None, note)
    elif setup in ("C1", "C2", "C3", "C-types"):
        if setup == "C-types":
            index = [{"group": e["name"], "description": e["description"], "applies_when": e["applies_when"]} for e in TYPES]
            tool = TYPE_TOOL
        else:
            index = [entry(g, int(setup[1])) for g in GROUPS]
            tool = SEARCH_TOOL
        text = ("You are a helpful assistant.\n\n" + C_INSTRUCTION + "\n\nMemory index:\n"
                + json.dumps(index, ensure_ascii=False, indent=1))
        st["system"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        if note:
            st["system"].append({"type": "text", "text": note.strip()})
        st["tools"] = [tool] + task_defs
        st["max_rounds"] = 6
        st["searchable"] = True
        ctx = None
        st["info"]["groups_searched"] = []
        st["seen"] = []
    else:
        raise ValueError(setup)
    if ctx is not None:
        st["ctx"] = summarize_ctx(ctx)
    return st


def handle_search(st, block):
    """Run a memory search for a C setup: the shared search inside the chosen group or type."""
    t = st["t"]
    item = ITEMS[t["item"]]
    rows = corpus(item, t["shuffle"], t["version"])
    position = {r["key"]: i for i, r in enumerate(rows)}
    g = block.get("input", {}).get("group")
    if t["setup"] == "C-types":
        members = [r for r in rows if TYPE_OF_GROUP.get(r["group"]) == g]
    else:
        members = [r for r in rows if r["group"] == g]
    top, _ = search.hybrid_search(item["query"], members)
    shown = place(top, t["depth"], position)
    st["seen"].extend({"gold": r["gold"], "secondary": r.get("secondary", False), "id": r["id"]} for r in shown)
    st["info"]["groups_searched"].append(g)
    return config.MEMORY_INSTRUCTION + "\n\n" + json.dumps(as_json(shown), ensure_ascii=False, indent=1)


def params(st):
    p = dict(model=st["t"]["model"], max_tokens=8000, system=st["system"],
             thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": "medium"},
             messages=st["messages"])
    if st["tools"]:
        p["tools"] = st["tools"]
    return p


def apply(st, msg):
    """Update a trial's state with one response (a dict, as from model_dump)."""
    st["round"] += 1
    u = msg.get("usage") or {}
    st["usage"].append({k: u.get(k) or 0 for k in ("input_tokens", "output_tokens",
                                                   "cache_read_input_tokens", "cache_creation_input_tokens")})
    st["stop"] = msg.get("stop_reason")
    content = msg.get("content") or []
    st["text"] += "".join(b.get("text", "") for b in content if b.get("type") == "text")
    st["thinking"] += "\n".join(b.get("thinking") or "" for b in content if b.get("type") == "thinking")
    calls = [b for b in content if b.get("type") == "tool_use"]
    if not calls or st["round"] >= st["max_rounds"]:
        st["done"] = True
        return
    results = []
    for b in calls:
        st["trace"].append(b["name"])
        out = handle_search(st, b) if b["name"] == SEARCH_TOOL["name"] else harness.STUB_RESULT
        results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out})
    clean = [{k: v for k, v in b.items() if v is not None and k != "citations"} for b in content]
    st["messages"] = st["messages"] + [{"role": "assistant", "content": clean}, {"role": "user", "content": results}]


def finalize(st):
    t = st["t"]
    item = ITEMS[t["item"]]
    if st.get("searchable"):
        seen = st["seen"]
        ctx = {"gold_in_context": any(r["gold"] for r in seen), "secondary_in_context": sum(r["secondary"] for r in seen),
               "n_memories_in_context": len(seen)}
    else:
        ctx = st["ctx"]
    task_trace = [x for x in st["trace"] if x != SEARCH_TOOL["name"]]
    res = {"reply": st["text"], "thinking": st["thinking"], "tool_trace": task_trace,
           "first_tool": task_trace[0] if task_trace else None, "stop_reason": st["stop"]}
    tot = {k: sum(u[k] for u in st["usage"]) for k in st["usage"][0]} if st["usage"] else {}
    info = dict(st["info"])
    gold_text = item.get(f"{t['version']}_memory")
    if gold_text and "groups_searched" in info:
        g = ONTO["gold_filing"][gold_text]
        info["gold_group"] = TYPE_OF_GROUP[g] if t["setup"] == "C-types" else g
        info["gold_group_searched"] = info["gold_group"] in info["groups_searched"]
    g = grade(item, res, t["version"]) if st["stop"] != "refusal" else {"passes": False, "method": "refusal", "judge_tokens": None}
    row = {**t, "scenario": item["scenario"], **ctx,
           "any_gold_in_context": ctx["gold_in_context"] or ctx["secondary_in_context"] > 0,
           "n_secondary": len(item.get("secondary_gold") or []), **info, **res, **tot, "rounds": st["round"],
           "passes": g["passes"], "grade_method": g["method"], "judge_tokens": g["judge_tokens"]}
    if ctx["gold_in_context"] and t["version"] == "gold":
        row["critic"] = critic(item, res, gold_text)
    return row


# --- executing ---------------------------------------------------------------------

def to_dict(msg):
    return json.loads(msg.model_dump_json(exclude_none=True))


def run_standard(states, workers):
    def go(st):
        while not st["done"]:
            st_params = params(st)
            r = harness.client().messages.create(**st_params)
            apply(st, to_dict(r))
        return st
    return harness.pmap(go, states, workers=workers)


def run_batched(states, max_requests=2000, max_bytes=150_000_000):
    """One batch per round (split by size); memory searches and stub results run locally."""
    client = harness.client()
    while True:
        active = [s for s in states if not s["done"]]
        if not active:
            return states
        chunks, cur, size = [], [], 0
        for s in active:
            p = params(s)
            n = len(json.dumps(p))
            if cur and (len(cur) >= max_requests or size + n > max_bytes):
                chunks.append(cur)
                cur, size = [], 0
            cur.append((s, p))
            size += n
        if cur:
            chunks.append(cur)
        batches = []
        for ci, chunk in enumerate(chunks):
            reqs = [{"custom_id": f"r{i}", "params": p} for i, (_, p) in enumerate(chunk)]
            b = client.messages.batches.create(requests=reqs)
            batches.append((b.id, chunk))
            print(f"  round {active[0]['round'] + 1}: batch {b.id} with {len(chunk)} requests", flush=True)
        for bid, chunk in batches:
            while client.messages.batches.retrieve(bid).processing_status != "ended":
                time.sleep(30)
            by_id = {}
            for res in client.messages.batches.results(bid):
                by_id[res.custom_id] = res
            for i, (s, _) in enumerate(chunk):
                res = by_id.get(f"r{i}")
                if res is None or res.result.type != "succeeded":
                    s["failed"] = getattr(res.result, "type", "missing") if res else "missing"
                    s["done"] = True
                    continue
                apply(s, to_dict(res.result.message))


# --- plans ---------------------------------------------------------------------------

HAIKU_SETUPS = ["A", "E", "B1", "G", "C1", "C2", "C3", "C-types", "C-embed", "C-Jev"]
SONNET_SETUPS = ["A", "E", "B1", "G", "C2", "C-types", "C-Jev"]
SWAP_SETUPS = ["A", "E", "B1", "G", "C2", "C-types", "C-Jev"]


def main_items():
    return sorted(i for i, it in ITEMS.items() if i not in DEV)


def sonnet_items():
    gold = [ITEMS[i] for i in main_items() if ITEMS[i]["gold_memory"]]
    by = {}
    for it in gold:
        by.setdefault(it["scenario"], []).append(it["id"])
    rng = random.Random(8)
    chosen = []
    for sc, ids in sorted(by.items()):
        k = round(100 * len(ids) / len(gold))
        chosen += rng.sample(sorted(ids), min(k, len(ids)))
    nm = [i for i in main_items() if not ITEMS[i]["gold_memory"]]
    return sorted(chosen) + rng.sample(sorted(nm), 20)


PLANS = {
    "main_haiku": dict(model="haiku", items=main_items, setups=HAIKU_SETUPS, depths=["top", "middle", "end"], version="gold"),
    "swap_haiku": dict(model="haiku", items=lambda: [i for i in main_items() if ITEMS[i]["gold_memory"]],
                       setups=SWAP_SETUPS, depths=["middle"], version="twin", baselines=False),
    "main_sonnet": dict(model="sonnet", items=sonnet_items, setups=SONNET_SETUPS, depths=["middle"], version="gold"),
    "jevplus_haiku": dict(model="haiku", items=main_items, setups=["C-Jev+"], depths=["top", "middle", "end"],
                          version="gold", baselines=False),
    "jevplus_sonnet": dict(model="sonnet", items=sonnet_items, setups=["C-Jev+"], depths=["middle"], version="gold",
                           baselines=False),
    "jevplus_swap_haiku": dict(model="haiku", items=lambda: [i for i in main_items() if ITEMS[i]["gold_memory"]],
                               setups=["C-Jev+"], depths=["middle"], version="twin", baselines=False),
    "llmpick_haiku": dict(model="haiku", items=main_items, setups=["L-cat", "L-all"], depths=["top", "middle", "end"],
                          version="gold", baselines=False),
    "llmpick_sonnet": dict(model="sonnet", items=sonnet_items, setups=["L-cat", "L-all"], depths=["middle"],
                           version="gold", baselines=False),
    "smoke_engine": dict(model="haiku", items=lambda: main_items()[::40], setups=HAIKU_SETUPS, depths=["middle"], version="gold"),
    "smoke_sonnet": dict(model="sonnet", items=lambda: sonnet_items()[::20], setups=SONNET_SETUPS, depths=["middle"], version="gold"),
}


def trials(plan):
    model = MODELS[plan["model"]]
    out = []
    for i in plan["items"]():
        it = ITEMS[i]
        sh = shuffle_of(i)
        base = dict(item=i, shuffle=sh, version=plan["version"], model=model)
        if plan.get("baselines", True):
            out.append({**base, "setup": "none", "depth": "none"})
            if it["gold_memory"]:
                out.append({**base, "setup": "D", "depth": "top"})
        for s in plan["setups"]:
            for d in (plan["depths"] if it["gold_memory"] else ["middle"]):
                out.append({**base, "setup": s, "depth": d})
    for t in out:
        t["key"] = f"{t['item']}|{t['setup']}|{t['depth']}|{t['shuffle']}|{t['version']}|{t['model']}"
    return out


TOKENS_K = {"none": 0.6, "D": 0.7, "A": 52, "E": 1.4, "B1": 11, "G": 19, "C1": 7, "C2": 9.5, "C3": 23,
            "C-types": 8, "C-embed": 2.7, "C-Jev": 8, "C-Jev+": 1.6, "L-cat": 1.2, "L-all": 1.3}


def estimate(ts, batch):
    """$ from pilot-measured input tokens per trial, ~1,000 output tokens, half of A's and the
    C indexes' input read from cache, plus ~$0.0006 per trial for judging and the critic."""
    total = 0.0
    for t in ts:
        p_in, p_out = PRICE[t["model"]]
        k = TOKENS_K[t["setup"]] * 1000
        cached = 0.5 if t["setup"] in ("A", "C1", "C2", "C3", "C-types") else 0.0
        inp = k * (1 - cached) * p_in + k * cached * p_in * 0.1
        total += (inp + 1000 * p_out) / 1e6 * (0.5 if batch else 1.0) + 0.0006
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", choices=PLANS)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ts = trials(PLANS[args.plan])
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{args.plan}.jsonl"
    done = {json.loads(l)["key"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [t for t in ts if t["key"] not in done]
    est = estimate(todo, args.batch)
    print(f"{args.plan}: {len(ts)} trials, {len(todo)} to run, estimated ${est:.2f} ({'batched' if args.batch else 'standard'})")
    if args.estimate or not todo:
        return
    if est > args.max_cost:
        raise SystemExit(f"Estimate ${est:.2f} exceeds --max-cost ${args.max_cost:.2f}; not starting.")

    print("preparing trials (search, Jev scoring)...", flush=True)
    states = [s for s in harness.pmap(lambda t: _safe(prepare, t), todo, workers=args.workers) if s]
    print(f"prepared {len(states)}; running model calls", flush=True)
    states = run_batched(states) if args.batch else run_standard(states, args.workers)
    ok = [s for s in states if not s.get("failed")]
    print(f"{len(ok)} trials answered ({len(states) - len(ok)} failed and will rerun next time); grading", flush=True)

    def fin(s):
        row = _safe(finalize, s)
        if row:
            with _lock, path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
    rows = [r for r in harness.pmap(fin, ok, workers=args.workers) if r]
    print(f"logged {len(rows)} trials to {path.name}")


def _safe(fn, x):
    try:
        return fn(x)
    except Exception as e:  # logged and skipped; the trial reruns next time
        key = x["key"] if isinstance(x, dict) and "key" in x else x.get("t", {}).get("key")
        print(f"  failed {key}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None


if __name__ == "__main__":
    main()
