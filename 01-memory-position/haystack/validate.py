"""Validate haystack candidates and freeze haystack_v1.

Checks, in order (each drop is recorded with its reason):
  1. exact duplicates
  2. shape: starts with "User ", one sentence, no semicolons
  3. banned-word lint
  4. token length 18-24 on the target model's tokenizer   (API: count_tokens)
  5. near-duplicates (TF-IDF cosine)
  6. similarity to the needle (TF-IDF cosine)
  7. LLM review: would this item change a response's structure or layout?  (API)

    uv run python -m haystack.validate              # all checks, then freeze
    uv run python -m haystack.validate --offline    # checks 1-3, 5-6 only; no freeze
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

CANDIDATES = config.DATA / "candidates.jsonl"
TOKEN_CACHE = config.DATA / "token_counts.json"
REVIEW_CACHE = config.DATA / "review_flags.json"
SEED = 20261123
MIN_POOL = 2_700  # items needed for one 64k-token memory block

BANNED_RE = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(w) for w in config.BANNED_WORDS) + r")(?![a-z])",
    re.IGNORECASE,
)


# --- checks 1-3: offline text checks -------------------------------------

def shape_problem(text: str) -> str | None:
    if not text.startswith("User "):
        return "does not start with 'User '"
    if ";" in text:
        return "semicolon"
    # A second sentence = terminal punctuation followed by a capital letter.
    # ("6:30 p.m. with" stays one sentence.)
    if not text.endswith(".") or re.search(r"[.!?]\s+[A-Z]", text):
        return "not exactly one sentence ending in a period"
    return None


# --- check 4: token length -----------------------------------------------

def count_tokens(texts: list[str]) -> dict[str, int]:
    """Tokens of each text alone, on the target model's tokenizer.

    count_tokens measures a whole user message, so we subtract the fixed
    message overhead, calibrated as count("a") - 1.
    """
    import anthropic

    cache = json.loads(TOKEN_CACHE.read_text()) if TOKEN_CACHE.exists() else {}
    client = anthropic.Anthropic(max_retries=6)

    def raw(text):
        return client.messages.count_tokens(
            model=config.TARGET_MODEL, messages=[{"role": "user", "content": text}]
        ).input_tokens

    overhead = raw("a") - 1
    todo = [t for t in texts if t not in cache]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for text, n in zip(todo, pool.map(raw, todo)):
            cache[text] = n - overhead
    cache["__needle__"] = raw(config.NEEDLE) - overhead
    TOKEN_CACHE.write_text(json.dumps(cache, indent=0))
    return cache


# --- checks 5-6: lexical similarity --------------------------------------

def similarity_drops(texts: list[str]) -> tuple[dict[str, str], dict[str, float]]:
    """Greedy near-duplicate removal, then needle similarity.

    TF-IDF over word 1-2-grams. Lexical, not semantic: it catches reworded
    copies and needle-like vocabulary, which is what the experiment needs
    controlled. Thresholds live in config.
    """
    if not texts:
        return {}, {}
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = vec.fit_transform(texts + [config.NEEDLE])
    items, needle = matrix[:-1], matrix[-1]

    drops: dict[str, str] = {}
    sims = cosine_similarity(items)
    kept: list[int] = []
    for i in range(len(texts)):
        if kept and sims[i, kept].max() >= config.NEAR_DUP_THRESHOLD:
            drops[texts[i]] = "near-duplicate"
        else:
            kept.append(i)

    needle_sims = cosine_similarity(items, needle).ravel()
    needle_by_text = {}
    for i, text in enumerate(texts):
        needle_by_text[text] = float(needle_sims[i])
        if text not in drops and needle_sims[i] >= config.NEEDLE_SIM_THRESHOLD:
            drops[text] = "too similar to needle"
    return drops, needle_by_text


# --- check 7: LLM review -------------------------------------------------

class Flag(BaseModel):
    index: int
    reason: str


class Review(BaseModel):
    flagged: list[Flag]


REVIEW_PROMPT = """Below are numbered user preferences that will sit in an AI \
assistant's memory during an experiment about response formatting.

Flag every item that could plausibly change the STRUCTURE, LAYOUT, LENGTH or \
TONE of the assistant's replies: tables, lists, headings, bullet points, \
paragraphs, code blocks, brevity, verbosity, formality, emoji, and so on.

Also flag every item that touches any of these behaviours, even indirectly:
- units of measurement or a unit system (metric, imperial, temperatures, distances, weights)
- spelling variety (British vs American English)
- asking the user questions, or how a reply should open or end

Do NOT flag items that only change other content conventions (currency, date \
formats, names, which examples to use) or that describe the user's life \
without touching the behaviours above.

Return only the flagged items, with a short reason each. Return an empty list \
if nothing qualifies.

{items}"""


# Planted problem items that dodge the banned-word list. Three are mixed into
# every review batch so the report shows the reviewer's catch rate; they are
# never stored or kept.
CANARIES = [
    "User likes replies kept to a sentence or two at most.",
    "User enjoys emoji sprinkled through chat messages.",
    "User prefers answers that start with the key takeaway.",
    "User likes a quick check-in at the end of every reply.",
    "User prefers distances in km for running routes.",
    "User prefers words like colour written the UK way.",
    "User likes code shown in separate blocks with comments.",
    "User likes a warm, casual voice in assistant replies.",
    "User wants the thermostat kept at 20°C in winter.",
    "User prefers recipes with oven temperatures given in gas marks.",
]
CANARY_LOG = config.DATA / "review_canaries.json"


def review(texts: list[str]) -> dict[str, str]:
    import anthropic

    cache = json.loads(REVIEW_CACHE.read_text()) if REVIEW_CACHE.exists() else {}
    client = anthropic.Anthropic(max_retries=6)
    todo = [t for t in texts if t not in cache]
    canary_log = json.loads(CANARY_LOG.read_text()) if CANARY_LOG.exists() else {"planted": 0, "caught": 0}
    rng = random.Random(SEED)
    for start in range(0, len(todo), 100):
        real = todo[start:start + 100]
        planted = rng.sample(CANARIES, 3)
        chunk = real + planted
        rng.shuffle(chunk)
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk))
        response = client.messages.parse(
            model=config.GENERATOR_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(items=numbered)}],
            output_format=Review,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Review refused: {response.stop_details}")
        flags = {f.index: f.reason for f in response.parsed_output.flagged}
        for i, text in enumerate(chunk):
            if text in planted:
                canary_log["planted"] += 1
                canary_log["caught"] += i in flags
            else:
                cache[text] = flags.get(i, "")
        REVIEW_CACHE.write_text(json.dumps(cache, indent=0))
        CANARY_LOG.write_text(json.dumps(canary_log))
        print(f"reviewed {min(start + 100, len(todo))}/{len(todo)}")
    return {t: r for t, r in cache.items() if t in set(texts) and r}


# --- freeze --------------------------------------------------------------

def freeze(rows: list[dict], tokens: dict[str, int], report: dict) -> None:
    rng = random.Random(SEED)
    ids = rng.sample(range(16 ** 4), len(rows) + 1)
    needle_id = f"{ids[-1]:04x}"

    items = [{"id": f"{ids[i]:04x}", "preference": r["text"]} for i, r in enumerate(rows)]
    meta = [
        {"id": items[i]["id"], "category": r["category"], "kind": r["kind"], "tokens": tokens[r["text"]]}
        for i, r in enumerate(rows)
    ]
    blob = json.dumps(items, indent=1, ensure_ascii=False).encode()
    digest = hashlib.sha256(blob).hexdigest()

    (config.DATA / "haystack_v1.json").write_bytes(blob)
    (config.DATA / "haystack_v1.meta.json").write_text(json.dumps({
        "sha256": digest,
        "target_model": config.TARGET_MODEL,
        "needle": {"id": needle_id, "preference": config.NEEDLE, "tokens": tokens["__needle__"]},
        "items": meta,
    }, indent=1))
    report["sha256"] = digest
    print(f"\nFrozen {len(items)} items -> data/haystack_v1.json  sha256={digest[:16]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="skip API checks; do not freeze")
    parser.add_argument("--candidates", default=str(CANDIDATES), help="candidates file to read")
    args = parser.parse_args()

    from pathlib import Path
    rows = [json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l.strip()]
    total = len(rows)
    unique, seen = [], set()
    for r in rows:
        if r["text"].lower() not in seen:
            seen.add(r["text"].lower())
            unique.append(r)
    exact_dups = total - len(unique)
    rows = unique

    drops: dict[str, str] = {}
    for r in rows:
        t = r["text"]
        if problem := shape_problem(t):
            drops[t] = f"shape: {problem}"
        elif m := BANNED_RE.search(t):
            drops[t] = f"banned word: {m.group(0).lower()}"

    alive = lambda: [r for r in rows if r["text"] not in drops]  # noqa: E731

    tokens: dict[str, int] = {}
    if not args.offline:
        tokens = count_tokens([r["text"] for r in alive()])
        lo, hi = config.MIN_TOKENS, config.MAX_TOKENS
        for r in alive():
            if not lo <= tokens[r["text"]] <= hi:
                drops[r["text"]] = f"tokens: {tokens[r['text']]}"

    sim_drops, needle_sim = similarity_drops([r["text"] for r in alive()])
    drops.update(sim_drops)

    if not args.offline:
        for text, reason in review([r["text"] for r in alive()]).items():
            drops[text] = f"review: {reason}"

    final = alive()
    reasons = Counter(re.sub(r":.*", "", d) for d in drops.values())
    if exact_dups:
        reasons["exact duplicate"] = exact_dups
    report = {
        "candidates": total,
        "kept": len(final),
        "dropped_by_reason": dict(reasons.most_common()),
        "kept_by_category": dict(Counter(r["category"] for r in final)),
        "output_neutral_share": round(
            sum(r["kind"] == "output_neutral" for r in final) / max(len(final), 1), 3
        ),
        "max_needle_similarity": round(max((needle_sim[r["text"]] for r in final), default=0), 3),
        "offline": args.offline,
    }
    if CANARY_LOG.exists():
        report["review_canaries"] = json.loads(CANARY_LOG.read_text())
    if tokens:
        lengths = [tokens[r["text"]] for r in final]
        report["token_length"] = {
            "min": min(lengths), "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "needle": tokens["__needle__"],
        }

    (config.DATA / "drops.jsonl").write_text(
        "\n".join(json.dumps({"text": t, "reason": r}) for t, r in drops.items())
    )

    if not args.offline:
        if len(final) < MIN_POOL:
            print(f"Only {len(final)} items survive; need {MIN_POOL}. Generate more, then rerun.")
        elif not config.MIN_TOKENS <= tokens["__needle__"] <= config.MAX_TOKENS:
            print(f"Needle is {tokens['__needle__']} tokens, outside the item window. Adjust before freezing.")
        else:
            freeze(final, tokens, report)

    (config.DATA / "validation_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
