"""One-question smoke test: run a single query with the needle at a few depths.

Not the experiment: one sample per condition, no statistics. It checks that
the pieces fit together and shows what a trial looks like end to end.

    uv run python -m harness.smoke "How do I keep basil alive indoors?"
"""

import json
import random
import re
import sys

import anthropic

from haystack import config

MODEL = config.TARGET_MODEL
N_ITEMS = 60  # ~2k tokens of memory
SEED = 1

SYSTEM = """You are a helpful assistant. Below are memories you have stored about \
this user from earlier conversations, as JSON. Use them when they are relevant.

<memories>
{memories}
</memories>"""

# A Markdown table: a header row followed by a |---|---| separator row.
TABLE_RE = re.compile(r"^\s*\|.*\|\s*\n\s*\|?\s*:?-{3,}", re.MULTILINE)


def memory_block(depth: float | None) -> tuple[str, int | None]:
    items = json.load(open(config.DATA / "haystack_v1.json"))
    meta = json.load(open(config.DATA / "haystack_v1.meta.json"))
    rng = random.Random(SEED)
    chosen = rng.sample(items, N_ITEMS)  # same haystack in every condition
    position = None
    if depth is not None:
        needle = {"id": meta["needle"]["id"], "preference": meta["needle"]["preference"]}
        position = round(depth * len(chosen))
        chosen = chosen[:position] + [needle] + chosen[position:]
    return json.dumps(chosen, indent=1, ensure_ascii=False), position


def run(query: str, depth: float | None, client) -> dict:
    memories, position = memory_block(depth)
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        output_config={"effort": "medium"},
        system=SYSTEM.format(memories=memories),
        messages=[{"role": "user", "content": query}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return {
        "depth": "no needle" if depth is None else f"{depth:.0%}",
        "position": position,
        "input_tokens": response.usage.input_tokens,
        "table": bool(TABLE_RE.search(text)),
        "stop_reason": response.stop_reason,
        "reply": text,
    }


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How do I keep basil alive indoors?"
    client = anthropic.Anthropic()
    results = [run(query, d, client) for d in (None, 0.0, 0.5, 1.0)]
    out = config.DATA / "smoke_results.json"
    out.write_text(json.dumps({"query": query, "model": MODEL, "results": results}, indent=1))
    print(f"Query: {query}\n")
    for r in results:
        print(f"{r['depth']:>9} | item #{r['position']} | {r['input_tokens']} input tokens | "
              f"table: {'YES' if r['table'] else 'no'} | {r['stop_reason']}")
    print(f"\nFull replies saved to {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
