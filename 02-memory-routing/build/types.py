"""Type-level ontology for C-types: one entry per memory type, written from its groups.

    uv run python -m build.types      # writes data/ontology_types.json

Opus sees only the group entries of each type (no questions) and writes a name,
description and "applies when" for the type as a whole. Every memory's type is
its group's type, so filing is inherited from the topic ontology.
"""

import json

from pydantic import BaseModel

from . import config, harness

OUT = config.DATA / "ontology_types.json"
TYPE_NAMES = {"style rules": "Style rules", "task procedures": "Task procedures",
              "facts": "Facts", "past sessions": "Past sessions"}


class TypeEntry(BaseModel):
    description: str
    applies_when: str


def main():
    onto = json.loads((config.DATA / "ontology_v1.json").read_text())
    out = []
    for t, name in TYPE_NAMES.items():
        groups = [g for g in onto["groups"] if g["type"] == t]
        listing = "\n".join(f"- {g['name']}: {g['description']} Applies when: {g['applies_when']}" for g in groups)
        r = harness.client().messages.parse(
            model=config.GENERATOR_MODEL, max_tokens=4000, output_config={"effort": "low"},
            system=("You write the index entry for one type of an AI assistant's long-term memory about a "
                    "user. From the topic groups of that type, write: description (one sentence on what "
                    "memories of this type cover) and applies_when (one sentence on when an assistant "
                    "should consult this type before replying, judged by what the memories are for)."),
            messages=[{"role": "user", "content": f"Type: {name}\nIts groups:\n{listing}"}],
            output_format=TypeEntry)
        out.append({"name": name, "type": t, "memory_type": groups[0]["memory_type"],
                    "description": r.parsed_output.description, "applies_when": r.parsed_output.applies_when,
                    "size": sum(g["size"] for g in groups), "groups": [g["name"] for g in groups]})
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for e in out:
        print(f"{e['name']} ({e['size']}): {e['applies_when']}")


if __name__ == "__main__":
    main()
