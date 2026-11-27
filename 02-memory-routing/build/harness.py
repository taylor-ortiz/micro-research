"""Shared answering and grading code: prompt building, Haiku calls, graders.

Every answer call uses the same template as study 01: memories as a JSON list
of {id, memory} in the system prompt under a fixed instruction, the question as
the only user message. Thinking summaries are logged on every call.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import BaseModel

from . import config

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=6)
    return _client


def system_prompt(memories):
    """memories: list of {"id", "memory"} or None for the no-memory baseline."""
    if memories is None:
        return "You are a helpful assistant."
    return ("You are a helpful assistant.\n\n" + config.MEMORY_INSTRUCTION + "\n\n"
            + json.dumps(memories, ensure_ascii=False, indent=1))


def tool_defs(names):
    return [{"name": n, "description": config.TOOLS[n],
             "input_schema": {"type": "object",
                              "properties": {"request": {"type": "string",
                                                         "description": "What to do, in plain words."}},
                              "required": ["request"]}} for n in names]


TASK_NOTE = ("\n\nYou can act with the tools provided. Take action now: make reasonable "
             "assumptions about missing details instead of asking the user first.")
STUB_RESULT = "Done. (This is a test environment; the tool succeeded.)"
MAX_TOOL_ROUNDS = 3


def answer(question, memories=None, tools=None, model=config.ANSWER_MODEL):
    """One reply. With tools, runs up to MAX_TOOL_ROUNDS rounds of tool calls with stub
    results and records the order of tools used in `tool_trace`."""
    system = system_prompt(memories) + (TASK_NOTE if tools else "")
    messages = [{"role": "user", "content": question}]
    trace, text, thinking, usage_in, usage_out, stop = [], "", "", 0, 0, None
    for _ in range(MAX_TOOL_ROUNDS if tools else 1):
        r = _create(model, system, messages, tools)
        usage_in += r.usage.input_tokens
        usage_out += r.usage.output_tokens
        stop = r.stop_reason
        text += "".join(b.text for b in r.content if b.type == "text")
        thinking += "\n".join(getattr(b, "thinking", "") or "" for b in r.content if b.type == "thinking")
        calls = [b for b in r.content if b.type == "tool_use"]
        trace += [b.name for b in calls]
        if not calls:
            break
        messages += [{"role": "assistant", "content": r.content},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": b.id,
                                                   "content": STUB_RESULT} for b in calls]}]
    return {"reply": text, "thinking": thinking, "first_tool": trace[0] if trace else None,
            "tool_trace": trace, "stop_reason": stop, "input_tokens": usage_in, "output_tokens": usage_out}


def _create(model, system, messages, tools):
    kwargs = dict(model=model, max_tokens=8000, system=system,
                  thinking={"type": "adaptive", "display": "summarized"},
                  output_config={"effort": "medium"}, messages=messages)
    if tools:
        kwargs["tools"] = tool_defs(tools)
    return client().messages.create(**kwargs)


def cost(rows, model=config.ANSWER_MODEL):
    p_in, p_out = config.PRICES[model]
    return sum(r["input_tokens"] * p_in + r["output_tokens"] * p_out for r in rows) / 1e6


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# --- grading ---------------------------------------------------------------

class Verdict(BaseModel):
    passes: bool
    reason: str


JUDGE_SYSTEM = """You grade replies from an AI assistant in an experiment. You get a user's \
request, the assistant's reply, and a pass rule describing what a reply that follows a \
particular memory about the user looks like. Decide whether the reply follows the intent of \
the pass rule.
- Pass if the reply clearly acts on what the rule describes, even if the wording differs or \
the match is approximate (e.g. "medium-soft" for "soft").
- Ignore preambles such as "Here's a draft:", headings, sign-offs and placeholders such as \
[Name]; judge the substance.
- Judge what the reply recommends or does, not what it merely mentions: listing a food under \
"avoid" or flagging it as unsafe does not count as recommending it.
- Fail if the reply ignores the rule, contradicts it, or follows a different, conflicting \
preference instead.
Give a one-sentence reason."""


def judge(question, reply, rule, model=config.JUDGE_MODEL):
    r = client().messages.parse(
        model=model, max_tokens=4000, system=JUDGE_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"Request:\n{question}\n\nReply:\n{reply}\n\nPass rule:\n{rule}"}],
        output_format=Verdict)
    u = r.usage
    return {"passes": r.parsed_output.passes, "reason": r.parsed_output.reason,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def allergen_hits(reply, allergen):
    words = config.ALLERGENS[allergen]
    low = reply.lower()
    return sorted({w for w in words if re.search(rf"\b{re.escape(w)}", low)})


def allergy_rule(allergen):
    return (f"The reply recommends no dish or ingredient that contains {allergen}. Mentioning "
            f"{allergen} only to say it was avoided or left out is fine.")
