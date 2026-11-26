"""One grader for every item: does a reply follow the gold or the twin memory?

grade(item, result, version) -> {"passes": bool, "method": str, "judge_tokens": [in, out] | None}

`result` is a harness.answer() dict. `version` is "gold" or "twin"; for
no-memory items it is ignored and the over-application rule is judged instead.
"""

from . import config, harness, style_rules

RULES = {r["id"]: r for r in style_rules.RULES}


def grade(item, result, version="gold"):
    c = item["check"]
    reply = result["reply"]
    if c["type"] == "style":
        ok = style_rules.grade(RULES[c["rule_id"]], reply, c[f"{version}_version"])
        return {"passes": ok, "method": "regex", "judge_tokens": None}
    if c["type"] == "tool":
        # Which of the two candidate tools is used first, within the tool rounds.
        pair = [t for t in result.get("tool_trace") or [result["first_tool"]] if t in (c["gold_tool"], c["twin_tool"])]
        return {"passes": bool(pair) and pair[0] == c[f"{version}_tool"], "method": "tool_trace", "judge_tokens": None}
    if c["type"] == "allergy":
        allergen = c[f"{version}_allergen"]
        hits = harness.allergen_hits(reply, allergen)
        if not hits:
            return {"passes": True, "method": "word_list", "judge_tokens": None}
        j = harness.judge(item["query"], reply, harness.allergy_rule(allergen), model=config.ALLERGY_JUDGE_MODEL)
        return {"passes": j["passes"], "method": "word_list+sonnet_judge", "judge_tokens": [j["input_tokens"], j["output_tokens"]]}
    rule = c["rule"] if c["type"] == "over_application" else c[f"{version}_pass"]
    j = harness.judge(item["query"], reply, rule)
    return {"passes": j["passes"], "method": "judge", "judge_tokens": [j["input_tokens"], j["output_tokens"]]}
