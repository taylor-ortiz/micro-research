"""Generate haystack candidates with Claude.

Writes data/candidates.jsonl, one candidate per line. Resumable: rerunning
tops each category up to its target instead of starting over.

    uv run python -m haystack.generate                 # full pool (~3,500)
    uv run python -m haystack.generate --pilot         # 1 batch per category
"""

import argparse
import json
from collections import defaultdict

import anthropic
from pydantic import BaseModel

from . import config

CANDIDATES = config.DATA / "candidates.jsonl"

SYSTEM = """You write synthetic user preferences for a memory-retrieval experiment.

Each preference is one plain sentence about a fictional user, in third person, \
starting with "User prefers", "User likes", "User usually", "User enjoys", \
"User tends to" or "User avoids".

Rules for every sentence:
- 8 to 10 words in total, including the opening \"User prefers\". One preference only.\n- Plain everyday words. Avoid digits, clock times, symbols, abbreviations and rare names: they cost extra length.
- No gendered pronouns: write \"their\" or rephrase. No lists, no semicolons.
- Concrete and specific (a brand type, a time, a place, a quantity), not generic.
- No personal data: no real names of people, addresses, phone numbers or employers.
- No negation-heavy phrasing, no conditions that contradict other preferences.
- Never mention how answers, responses or text should be laid out or how long \
they should be, how they should open or end, units of measurement, or spelling. Avoid these words entirely: {banned}.
- Do not repeat or lightly reword any sentence in the "already written" list."""

INERT_NOTE = (
    "These preferences describe the user's life. They must NOT tell an assistant "
    "how to write its replies."
)
NEUTRAL_NOTE = (
    "These preferences DO tell an assistant something about its replies, but only "
    "about content conventions (currency, date and time conventions, naming, "
    "which examples or sources to use), never about layout, structure, tone or "
    "length, and never about units of measurement, spelling variety, or asking "
    "the user questions. Example: 'User prefers prices in euros for European "
    "travel budgets.'"
)


class Batch(BaseModel):
    preferences: list[str]


def load_existing() -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    if CANDIDATES.exists():
        for line in CANDIDATES.read_text().splitlines():
            row = json.loads(line)
            by_cat[row["category"]].append(row["text"])
    return by_cat


def generate_batch(client, category, description, kind, already, n):
    note = INERT_NOTE if kind == "inert" else NEUTRAL_NOTE
    avoid = "\n".join(f"- {t}" for t in already[-120:]) or "(none yet)"
    prompt = (
        f"Category: {category} ({description}).\n{note}\n\n"
        f"Write {n} new, varied preferences for this category.\n\n"
        f"Already written:\n{avoid}"
    )
    response = client.messages.parse(
        model=config.GENERATOR_MODEL,
        max_tokens=16000,
        system=SYSTEM.format(banned=", ".join(config.BANNED_WORDS)),
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Batch,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Refused for {category}: {response.stop_details}")
    return [p.strip() for p in response.parsed_output.preferences if p.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="one small batch per category")
    parser.add_argument("--only", nargs="+", help="limit to these categories")
    parser.add_argument("--count", type=int, help="target per category (overrides config)")
    args = parser.parse_args()

    config.DATA.mkdir(exist_ok=True)
    client = anthropic.Anthropic()
    existing = load_existing()

    plan = [(c, d, "inert", config.CANDIDATES_PER_INERT_CATEGORY)
            for c, d in config.INERT_CATEGORIES.items()]
    plan += [(c, d, "output_neutral", config.CANDIDATES_PER_NEUTRAL_CATEGORY)
             for c, d in config.OUTPUT_NEUTRAL_CATEGORIES.items()]

    with CANDIDATES.open("a") as out:
        for category, description, kind, target in plan:
            if args.only and category not in args.only:
                continue
            have = existing[category]
            goal = len(have) + 10 if args.pilot else (args.count or target)
            while len(have) < goal:
                n = min(config.BATCH_SIZE, goal - len(have))
                batch = generate_batch(client, category, description, kind, have, n)
                if not batch:
                    raise RuntimeError(f"Empty batch for {category}; stopping to avoid a loop")
                for text in batch:
                    out.write(json.dumps({"category": category, "kind": kind, "text": text}) + "\n")
                out.flush()
                have.extend(batch)
                print(f"{category:16s} {len(have):4d}/{goal}")


if __name__ == "__main__":
    main()
