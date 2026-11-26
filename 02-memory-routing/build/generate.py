"""Generate gold memories, swap twins, questions and distractors with Claude.

Writes one JSONL file per kind under data/gen/. Resumable: rerunning tops each
kind up to its target instead of starting over.

    uv run python -m build.generate --pilot          # 5 items per kind, for review
    uv run python -m build.generate                  # full targets
    uv run python -m build.generate --only fact allergy
"""

import argparse
import json
from typing import Literal

import anthropic
from pydantic import BaseModel

from . import config

OUT = config.DATA / "gen"
BATCH = 10

COMMON = """You write synthetic test material for an experiment on how AI assistants use \
stored memories about a fictional user. Memories are short third-person sentences \
starting with "User". Rules:
- No real people's names, addresses, phone numbers or employers.
- No gendered pronouns: write "their" or rephrase.
- Every item must be clearly different from the items already written (listed below).
- Plain, natural wording, as a real assistant's memory store would hold it."""

PAIR_RULES = """
For each item write:
- query: a natural request a user would send, which does NOT mention the memory.
- gold_memory: one sentence (10 to 18 words) that should clearly change a good answer.
- twin_memory: a minimal edit of gold_memory that should change the answer in a different, \
checkable direction (e.g. vegetarian vs pescatarian, window vs aisle seat).
- gold_pass / twin_pass: one sentence each, saying what a reply must do to count as \
following that version. Make them observable in the reply text, and mutually exclusive.
Without any memory, a typical answer should satisfy neither pass rule."""

KINDS = {
    "fact": dict(target=40, system=COMMON + PAIR_RULES, prompt=(
        "Kind: semantic facts about the user's life (diet, family, home, location, hobbies, "
        "health limits other than food allergies, work, pets, budget). The query is an everyday "
        "request where a helpful assistant would use the fact. Do not use food allergies.")),
    "unclear_type": dict(target=70, system=COMMON + PAIR_RULES, prompt=(
        "Kind: memories whose type is ambiguous: a fact about the user that acts like a standing "
        "rule for one kind of task (e.g. 'User prefers aisle seats on flights over three hours' "
        "for a flight-planning query). The query is a task where the memory should be applied, "
        "answered in text (no tools). Do not use food allergies, diets or foods.")),
    "past_session": dict(target=40, system=COMMON + PAIR_RULES, prompt=(
        "Kind: episodic memories. gold_memory is a dated summary of a past session, one or two "
        "sentences, starting 'On 2026-MM-DD' (dates between 2026-01-05 and 2026-11-20), e.g. "
        "'On 2026-10-14 we debugged a token-refresh bug in the auth service and left the retry "
        "logic unfinished.' The query refers back to that session without restating it ('Where "
        "did we leave the auth bug?'). twin_memory changes the one key detail the answer depends "
        "on. Mix work, study and personal projects. gold_memory may be up to 30 words.")),
    "allergy": dict(target=55, system=COMMON, prompt=(
        "Kind: food-allergy queries. For each item give: allergen (one of: peanut, tree nut, "
        "shellfish, sesame, dairy), twin_allergen (a different one from that list that is ALSO "
        "common in the dish or cuisine), direct (true or false) and query. Indirect queries "
        "(direct=false) name a cuisine, dish or occasion where the allergen is usually present "
        "but never name the allergen or its food group (e.g. 'Plan a Thai takeout order for two'). "
        "Direct queries name the food group ('Which nuts make the best trail mix?'). Make about "
        "3 of every 4 items indirect. Every query must ask for specific dishes or ingredients.")),
    "task_procedure": dict(target=70, system=COMMON, prompt=(
        "Kind: task procedures for an assistant that has tools. For each item give: task_type "
        "(short label), query (a request to do the task, e.g. 'Find me a flight to Denver next "
        "Thursday'), target_tool and twin_tool (two DIFFERENT tools from the catalog below that "
        "could both plausibly start the task), other_tools (exactly 6 more catalog tools, not "
        "the target or twin), gold_memory (a procedure of 12 to 25 words saying to start with "
        "target_tool, e.g. 'For flights, User wants travel_portal searched first, then the "
        "calendar checked'), twin_memory (the same procedure starting with twin_tool instead). "
        "The query must not name any tool.\n\nTool catalog:\n"
        + "\n".join(f"- {k}: {v}" for k, v in config.TOOLS.items()))),
    "task_decoy": dict(target=config.N_TASK_DECOYS, system=COMMON, prompt=(
        "Kind: decoy task procedures, distractors in the memory store. Each memory is one "
        "procedure of 12 to 25 words about how the user wants one kind of task done, naming one "
        "or two tools from the catalog below (e.g. 'For sharing design files, User wants them "
        "uploaded with drive_upload and then linked in slack_post'). Vary task types widely. "
        "Do NOT cover any of these task types, which belong to other memories: {gold_tasks}.\n\nTool catalog:\n"
        + "\n".join(f"- {k}: {v}" for k, v in config.TOOLS.items()))),
    "universal_rule": dict(target=60, system=COMMON, prompt=(
        "Kind: standing rules about how the user wants EVERY reply written, whatever the topic, as "
        "distractors in the memory store. Each memory is one sentence of 10 to 16 words starting "
        "'User prefers' or 'User wants' (e.g. 'User prefers plain everyday words over technical "
        "jargon in every reply'). Vary them widely: word choice, sentence length, paragraph length, "
        "emphasis, punctuation, spelling of particular words, caveats, hedging. They must NOT touch: "
        "units or measurements, currencies, prices, dates, clock times, time zones, how to address "
        "or greet the user, how replies open or close or sign off, labels such as Step, Phase, "
        "Option or Choice, confidence or certainty statements, follow-up questions, P.S. or Note "
        "lines, emoji, tone (formal or casual), bullet points or summaries at the top, or anything "
        "about tools, tasks or a specific topic.")),
    "session_distractor": dict(target=config.N_EPISODIC + 100, system=COMMON, prompt=(
        "Kind: episodic distractors for the memory store. Each memory is a dated summary of a "
        "past session with the assistant, one or two sentences of 18 to 30 words, starting "
        "'On 2026-MM-DD we' and written in the first person plural (e.g. 'On 2026-04-18 we planned "
        "the user's backyard shed build; the frame was finished but the roof was not'), with "
        "dates between 2026-01-05 and 2026-11-20. Mix coding, writing, planning, "
        "study, home and hobby sessions. Do not mention food allergies, units, currencies, time "
        "zones or forms of address.")),
}


class PairItem(BaseModel):
    query: str
    gold_memory: str
    twin_memory: str
    gold_pass: str
    twin_pass: str


Allergen = Literal["peanut", "tree nut", "shellfish", "sesame", "dairy"]


class AllergyItem(BaseModel):
    allergen: Allergen
    twin_allergen: Allergen
    direct: bool
    query: str


class TaskItem(BaseModel):
    task_type: str
    query: str
    target_tool: str
    twin_tool: str
    other_tools: list[str]
    gold_memory: str
    twin_memory: str


class MemoryItem(BaseModel):
    memory: str


SCHEMA = {"fact": PairItem, "unclear_type": PairItem, "past_session": PairItem,
          "allergy": AllergyItem, "task_procedure": TaskItem,
          "task_decoy": MemoryItem, "session_distractor": MemoryItem,
          "universal_rule": MemoryItem}


def batch_model(item_cls):
    return type(f"{item_cls.__name__}Batch", (BaseModel,), {"__annotations__": {"items": list[item_cls]}})


def summary(kind, row):
    if kind == "allergy":
        return f"{row['allergen']} / {row['query']}"
    if kind in ("task_decoy", "session_distractor", "universal_rule"):
        return row["memory"]
    return f"{row['query']} -> {row['gold_memory']}"


def valid(kind, row):
    if kind == "allergy":
        return row["allergen"] != row["twin_allergen"]
    if kind == "task_procedure":
        tools = [row["target_tool"], row["twin_tool"], *row["other_tools"]]
        return (len(row["other_tools"]) == config.TOOLS_PER_TASK - 2 and len(set(tools)) == len(tools)
                and all(t in config.TOOLS for t in tools)
                and not any(t in row["query"] for t in config.TOOLS))
    return True


def prompt_for(kind, spec):
    if kind != "task_decoy":
        return spec["prompt"]
    path = OUT / "task_procedure.jsonl"
    if not path.exists():
        raise RuntimeError("Generate task_procedure before task_decoy")
    tasks = sorted({json.loads(l)["task_type"] for l in path.read_text().splitlines()})
    return spec["prompt"].replace("{gold_tasks}", ", ".join(tasks))


def generate_batch(client, kind, spec, already, n):
    avoid = "\n".join(f"- {s}" for s in already[-150:]) or "(none yet)"
    response = client.messages.parse(
        model=config.GENERATOR_MODEL,
        max_tokens=16000,
        system=spec["system"],
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"{prompt_for(kind, spec)}\n\nWrite {n} new items.\n\nAlready written:\n{avoid}"}],
        output_format=batch_model(SCHEMA[kind]),
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Refused for {kind}: {response.stop_details}")
    u = response.usage
    cost = (u.input_tokens * 4 + u.output_tokens * 20) / 1e6
    return [i.model_dump() for i in response.parsed_output.items], cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="5 items per kind")
    parser.add_argument("--only", nargs="+", choices=list(KINDS))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    total = 0.0
    for kind, spec in KINDS.items():
        if args.only and kind not in args.only:
            continue
        path = OUT / f"{kind}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        goal = min(5, spec["target"]) if args.pilot else spec["target"]
        rejected = 0
        with path.open("a") as out:
            while len(rows) < goal:
                n = min(BATCH, goal - len(rows))
                batch, cost = generate_batch(client, kind, spec, [summary(kind, r) for r in rows], n)
                total += cost
                if not batch:
                    raise RuntimeError(f"Empty batch for {kind}; stopping to avoid a loop")
                for row in batch[: goal - len(rows)]:
                    if not valid(kind, row):
                        rejected += 1
                        continue
                    out.write(json.dumps(row) + "\n")
                    rows.append(row)
                out.flush()
                if rejected > 3 * goal:
                    raise RuntimeError(f"Too many invalid items for {kind}")
        print(f"{kind:20s} {len(rows):4d}/{goal}  rejected {rejected}  running cost ${total:.2f}")


if __name__ == "__main__":
    main()
